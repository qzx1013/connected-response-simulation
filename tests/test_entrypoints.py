"""Check packaged imports and command-line parsers without launching stages."""

import importlib
import pkgutil
import subprocess
import sys
import residual_tn


def test_all_deferred_imports_are_bundled_or_public():
    import ast
    from pathlib import Path

    public = {
        "PIL",
        "cotengra",
        "matplotlib",
        "numpy",
        "quimb",
        "torch",
        "scipy",
        "opt_einsum",
        "networkx",
    }
    seen = set()
    for path in Path(residual_tn.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                seen.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                seen.add(node.module.split(".")[0])
    assert not seen - set(sys.stdlib_module_names) - public - {"residual_tn"}


def test_all_project_modules_import_without_path_bootstrap():
    before = list(sys.path)
    for module in pkgutil.walk_packages(
        residual_tn.__path__, residual_tn.__name__ + "."
    ):
        if module.name.endswith(".__main__"):
            continue
        importlib.import_module(module.name)
    assert sys.path == before


def test_stage_parsers_are_callable_from_another_directory(tmp_path):
    for name in ("fig2", "fig3", "fig4", "pauli_fullstate"):
        result = subprocess.run(
            [sys.executable, "-I", "-m", "residual_tn.experiments." + name, "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--residual-manifest" in result.stdout
