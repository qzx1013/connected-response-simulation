"""Experimental norm-BP batching with temporary zero padding.

Every branch retains its own actual bond dimensions. Padding exists only in
the norm-BP workspace; messages are cropped before the original projectors.
No branches share a bra, and no convergence tolerance is relaxed.
"""

import math
import torch
import torch.nn.functional as F
from residual_tn.peps import model as p


def opposite(*args):
    return p.core._contract_bp_opposite_pair(*args, batched_bra=True)


def solve(branches, plan, cfg, warm=None):
    B = len(branches)
    ref = next(iter(branches[0].values()))
    device = ref.device
    shapes = {
        site: [tuple(b[site].shape[1:]) for b in branches] for site in branches[0]
    }
    joined = {}
    for site, rows in shapes.items():
        target = tuple(max(shape[axis] for shape in rows) for axis in range(5))
        joined[site] = torch.cat(
            [
                F.pad(
                    b[site],
                    tuple(
                        x
                        for axis in reversed(range(5))
                        for x in (0, target[axis] - b[site].shape[axis + 1])
                    ),
                )
                for b in branches
            ]
        )
    bra = {k: v.conj() for k, v in joined.items()}
    messages = {}
    for site, direction in plan["directed_edges"]:
        axis = p.core.DIR_TO_AXIS[direction]
        d = joined[site].shape[axis + 1]
        value = torch.zeros((B, d, d), dtype=ref.dtype, device=device)
        for i in range(B):
            original = shapes[site][i][axis]
            candidate = None if warm is None else warm.get((site, direction))
            if candidate is not None and tuple(candidate.shape) == (original, original):
                candidate = candidate.to(device=device)
                value[i, :original, :original] = candidate / torch.linalg.norm(
                    candidate
                )
            else:
                value[i, :original, :original] = torch.eye(
                    original, dtype=ref.dtype, device=device
                ) / math.sqrt(original)
        messages[site, direction] = value
    site_plan = p.core._pauli_batched_bra_site_plan(plan)
    active = torch.ones(B, dtype=torch.bool, device=device)
    iterations = torch.zeros(B, dtype=torch.int64, device=device)
    residuals = torch.full((B,), float("inf"), dtype=torch.float64, device=device)

    def raw_site(site, entries):
        output = {}
        if cfg.reuse_opposite_bp_cavities:
            for pair in (("U", "D"), ("L", "R")):
                shared = opposite(joined, bra, messages, site, pair, plan)
                if shared is not None:
                    output.update(shared)
        for _, direction, equation, incoming in entries:
            if (site, direction) not in output:
                output[site, direction] = p.core._contract_bp_edge(
                    joined, bra, messages, site, equation, incoming, plan
                )
        return output

    for iteration in range(1, cfg.bp_max_iter + 1):
        step = torch.zeros(B, dtype=torch.float64, device=device)
        bad = torch.zeros(B, dtype=torch.bool, device=device)
        for site, entries in site_plan:
            raw = raw_site(site, entries)
            for _, direction, _, _ in entries:
                key = (site, direction)
                old = messages[key]
                value, invalid, _ = p.core._normalize_message(raw[key])
                bad |= invalid & active
                value = p.core._phase_align(value, old)
                updated, invalid, _ = p.core._normalize_message(
                    (1 - cfg.bp_damping) * value + cfg.bp_damping * old
                )
                bad |= invalid & active
                step = torch.maximum(
                    step, (updated - old).abs().reshape(B, -1).max(-1).values
                )
                messages[key] = torch.where(active[:, None, None], updated, old)
        if (
            iteration != 1
            and iteration % cfg.bp_residual_check_interval
            and iteration != cfg.bp_max_iter
        ):
            continue
        if bool(bad.any()):
            raise RuntimeError("Invalid padded BP map")
        candidates = active.clone()
        if (
            iteration != cfg.bp_max_iter
            and cfg.bp_step_residual_gate_factor is not None
        ):
            candidates &= step <= cfg.bp_step_residual_gate_factor * cfg.bp_tol
        if not bool(candidates.any()):
            continue
        strict = torch.zeros(B, dtype=torch.float64, device=device)
        for site, entries in site_plan:
            raw = raw_site(site, entries)
            for _, direction, _, _ in entries:
                key = (site, direction)
                value, invalid, _ = p.core._normalize_message(raw[key])
                bad |= invalid & active
                value = p.core._phase_align(value, messages[key])
                strict = torch.maximum(
                    strict,
                    p.core._projective_residual(value, messages[key]).to(torch.float64),
                )
        if bool(bad.any()):
            raise RuntimeError("Invalid padded strict BP map")
        converged = candidates & (strict < cfg.bp_tol)
        iterations[converged] = iteration
        residuals[converged] = strict[converged]
        active &= ~converged
        if not bool(active.any()):
            break
    if bool(active.any()):
        raise RuntimeError("Padded norm BP did not converge")
    results = []
    for i in range(B):
        cropped = {}
        for (site, direction), value in messages.items():
            d = shapes[site][i][p.core.DIR_TO_AXIS[direction]]
            cropped[site, direction] = value[i, :d, :d].contiguous()
        results.append(cropped)
    info = dict(
        iterations_per_branch=iterations.cpu().tolist(),
        residual_per_branch=residuals.cpu().tolist(),
        padded_peps_bytes=sum(v.numel() * v.element_size() for v in joined.values()),
        original_peps_bytes=sum(
            v.numel() * v.element_size() for b in branches for v in b.values()
        ),
    )
    return results, info
