"""Exact serial tensor-network contraction with conservative memory planning.

The total budget is a planning target, not a CUDA-wide hard guarantee.  A
separately configured PyTorch allocator cap catches unmodelled temporaries.
No singular values or slices are discarded.  This inference-only executor
restarts a failed contraction from slice zero, never double-counting a sum.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import time
from typing import Any

import torch


GIB = 1024**3


@dataclass(frozen=True)
class MemoryPolicy:
    total_bytes: int
    intermediate_bytes: int
    reserve_bytes: int = 8 * GIB
    safety_factor: float = 2.0
    max_slices: int = 262144
    max_retries: int = 6
    search_repeats: int = 64

    def validate(self):
        if self.total_bytes <= 0 or self.intermediate_bytes <= 0:
            raise ValueError("memory budgets must be positive")
        if not 0 <= self.reserve_bytes < self.total_bytes:
            raise ValueError("reserve must be nonnegative and below total budget")
        if not math.isfinite(self.safety_factor) or self.safety_factor < 1:
            raise ValueError("memory safety factor must be finite and >= 1")
        if self.max_slices < 1 or self.max_retries < 0 or self.search_repeats < 1:
            raise ValueError("invalid slicing/retry/search limits")


def _tree_peak(tree) -> int:
    # Cotengra 0.7.5 lacks the newer reorder_for_peak_size API.  Evaluate
    # the same child-order recurrence directly through its tree interface.
    # The final live-set traversal includes ALL inputs and each new output.
    if hasattr(tree, "reorder_for_peak_size"):
        tree.reorder_for_peak_size()
    else:
        peaks = {}
        for parent, left, right in tuple(tree.traverse()):
            sl, sr, sp = (int(tree.get_size(node)) for node in (left, right, parent))
            pl, pr = peaks.get(left, sl), peaks.get(right, sr)
            left_first = max(pl, sl + pr, sl + sr + sp)
            right_first = max(pr, sr + pl, sl + sr + sp)
            if right_first < left_first:
                tree.children[parent] = right, left
            peaks[parent] = min(left_first, right_first)
    live = sum(int(tree.get_size(node)) for node in tree.gen_leaves())
    peak = live
    for parent, left, right in tree.traverse():
        live += int(tree.get_size(parent))
        peak = max(peak, live)
        live -= int(tree.get_size(left)) + int(tree.get_size(right))
    return max(peak, int(tree.max_size()))


def fit_tree(
    tree,
    policy: MemoryPolicy,
    *,
    element_size: int,
    baseline_bytes: int,
    target_bytes: int,
):
    """Slice until both the single-tensor and concurrent estimates fit."""
    policy.validate()
    room = policy.total_bytes - baseline_bytes - policy.reserve_bytes
    if room <= 0:
        raise RuntimeError("resident tensors plus reserve already exceed memory budget")
    target = max(1, min(target_bytes, int(room / policy.safety_factor)) // element_size)
    candidate = tree.copy()
    for _ in range(64):
        if int(candidate.max_size()) > target:
            candidate = candidate.slice(
                target_size=target,
                minimize="size",
                allow_outer=False,
                max_repeats=32,
                inplace=False,
            )
        if int(candidate.nslices) > policy.max_slices:
            raise RuntimeError(
                f"bounded gloop needs {candidate.nslices} exact slices; "
                f"configured maximum is {policy.max_slices}; no slices discarded"
            )
        peak = _tree_peak(candidate)
        predicted = (
            baseline_bytes
            + policy.reserve_bytes
            + math.ceil(policy.safety_factor * peak * element_size)
        )
        if int(candidate.max_size()) <= target and predicted <= policy.total_bytes:
            return candidate, {
                "slices": int(candidate.nslices),
                "max_tensor_bytes": int(candidate.max_size()) * element_size,
                "tree_peak_bytes": peak * element_size,
                "resident_bytes": baseline_bytes,
                "predicted_total_bytes": predicted,
                "target_bytes": target * element_size,
            }
        if target == 1:
            break
        target = max(1, min(target // 2, int(candidate.max_size()) // 2))
    raise RuntimeError("no exact contraction plan fits the requested peak budget")


@torch.no_grad()
def serial_sum(tree, operands, *, progress=None):
    """Keep one slice result and one accumulator, on one CUDA stream."""
    if any(value.requires_grad for value in operands):
        raise ValueError("bounded gloop is an inference-only executor")
    device = operands[0].device
    total = None
    last_report = time.monotonic()
    for index in range(int(tree.nslices)):
        part = tree.contract_slice(operands, index, backend="torch", autojit=False)
        # Never mutate a slice result that might alias an input tensor.
        if total is None:
            total = part.clone()
        else:
            total.add_(part)
        del part
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if progress is not None and (
            index == 0
            or index + 1 == int(tree.nslices)
            or time.monotonic() - last_report >= 60
        ):
            progress(index + 1, int(tree.nslices))
            last_report = time.monotonic()
    return total


def _attempt_sum(tree, operands, *, progress=None):
    # Leave the exception scope before retrying: its traceback otherwise
    # retains tensors owned by the failed contractor and partial sum.
    try:
        return serial_sum(tree, operands, progress=progress), None
    except torch.OutOfMemoryError as error:
        return None, str(error)


def contract(operands, inputs, output, plan: dict[str, Any], cache: dict):
    """Opt-in entry point called by the existing gloop region builder."""
    import cotengra as ctg

    policy = MemoryPolicy(
        total_bytes=int(float(plan["gloop_peak_budget_gib"]) * GIB),
        intermediate_bytes=int(float(plan["gloop_memory_target_gib"]) * GIB),
        reserve_bytes=int(float(plan["gloop_peak_reserve_gib"]) * GIB),
        safety_factor=float(plan["gloop_peak_safety_factor"]),
        max_slices=int(plan["gloop_memory_max_slices"]),
        max_retries=int(plan["gloop_oom_max_retries"]),
        search_repeats=int(plan["gloop_memory_search_repeats"]),
    )
    policy.validate()
    audit = plan["gloop_audit"]
    device = operands[0].device
    item_bytes = operands[0].element_size()
    shapes = tuple(tuple(value.shape) for value in operands)
    key = (
        "bounded-v1",
        tuple(inputs),
        tuple(output),
        shapes,
        policy,
        str(plan["gloop_optimize"]),
        str(operands[0].dtype),
    )
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = sum(value.numel() * value.element_size() for value in operands)
    cached = cache.get(key)
    if cached is None:
        audit["expression_cache_misses"] += 1
        original = ctg.array_contract_tree(
            inputs=tuple(inputs),
            output=tuple(output),
            shapes=shapes,
            optimize=plan["gloop_optimize"],
        )
        original_max = int(original.max_size())
        original_peak = _tree_peak(original)
        audit["memory_max_original_intermediate_elements"] = max(
            audit["memory_max_original_intermediate_elements"], original_max
        )
        if (
            original_max * item_bytes > policy.intermediate_bytes
            or baseline
            + policy.reserve_bytes
            + policy.safety_factor * original_peak * item_bytes
            > policy.total_bytes
        ):
            optimizer = ctg.HyperOptimizer(
                methods=["greedy", "random-greedy"],
                minimize="size",
                max_repeats=policy.search_repeats,
                parallel=False,
                progbar=False,
            )
            other = ctg.array_contract_tree(
                inputs=tuple(inputs),
                output=tuple(output),
                shapes=shapes,
                optimize=optimizer,
            )
            if _tree_peak(other) < original_peak:
                original = other
            audit["memory_researched_expression_count"] += 1
        tree, info = fit_tree(
            original,
            policy,
            element_size=item_bytes,
            baseline_bytes=baseline,
            target_bytes=policy.intermediate_bytes,
        )
        if len(cache) >= int(plan["gloop_expression_cache_size"]):
            cache.pop(next(iter(cache)))
            audit["expression_cache_evictions"] += 1
        cache[key] = (tree, info)
    else:
        audit["expression_cache_hits"] += 1
        tree, old_info = cached
        tree, info = fit_tree(
            tree,
            policy,
            element_size=item_bytes,
            baseline_bytes=baseline,
            target_bytes=old_info["target_bytes"],
        )
        cache.pop(key)
        cache[key] = (tree, info)
    elapsed = time.perf_counter() - started
    audit["expression_build_seconds"] += elapsed
    audit["max_expression_build_seconds"] = max(
        audit["max_expression_build_seconds"], elapsed
    )
    audit["memory_planned_expression_count"] += 1

    def report(index, count):
        print(f"[bounded-gloop-slices] completed={index}/{count}", flush=True)

    for attempt in range(policy.max_retries + 1):
        print(
            f"[bounded-gloop-plan] loop={plan['gloop_size']} attempt={attempt} "
            f"slices={info['slices']} single_gib={info['max_tensor_bytes'] / GIB:.4f} "
            f"tree_peak_gib={info['tree_peak_bytes'] / GIB:.4f} "
            f"predicted_total_gib={info['predicted_total_bytes'] / GIB:.4f}",
            flush=True,
        )
        audit["memory_max_final_intermediate_elements"] = max(
            audit["memory_max_final_intermediate_elements"],
            info["max_tensor_bytes"] // item_bytes,
        )
        audit["memory_max_slice_count"] = max(
            audit["memory_max_slice_count"], info["slices"]
        )
        for name, value in (
            ("max_tree_peak_bytes", info["tree_peak_bytes"]),
            ("max_predicted_total_bytes", info["predicted_total_bytes"]),
        ):
            audit[name] = max(audit.get(name, 0), value)
        if info["slices"] > 1:
            audit["memory_sliced_expression_count"] += 1
        result, error = _attempt_sum(
            tree, operands, progress=report if info["slices"] > 1 else None
        )
        if error is None:
            cache[key] = (tree, info)
            return result
        audit["oom_retry_count"] += 1
        print(f"[bounded-gloop-retry] attempt={attempt} reason={error}", flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if attempt == policy.max_retries:
            raise RuntimeError(
                "bounded gloop exhausted exact slicing retries; background is reusable"
            )
        tree, info = fit_tree(
            tree,
            policy,
            element_size=item_bytes,
            baseline_bytes=baseline,
            target_bytes=max(
                item_bytes, min(info["target_bytes"], info["max_tensor_bytes"]) // 2
            ),
        )
    raise AssertionError("unreachable")
