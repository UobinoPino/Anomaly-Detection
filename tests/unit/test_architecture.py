"""The layering rules, enforced.

Structure decays quietly, so the rules that prevent that recurring are tests
rather than prose:

  core  <-  data, backbones  <-  detectors, postprocess  <-  stacking, runner
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "spacepresso"

#: What each layer is allowed to import from, transitively closed.
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "config": {"core", "postprocess"},
    "data": {"core"},
    "backbones": {"core"},
    "detectors": {"core", "config", "data", "backbones", "postprocess"},
    "postprocess": {"core"},
    "stacking": {"core", "config", "data", "backbones", "postprocess"},
    "runner": {
        "core",
        "config",
        "data",
        "backbones",
        "detectors",
        "postprocess",
        "stacking",
    },
    "analysis": {
        "core",
        "config",
        "data",
        "backbones",
        "detectors",
        "postprocess",
        "stacking",
    },
}


def _modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def _layer_of(path: Path) -> str:
    relative = path.relative_to(SRC)
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


def _imports(path: Path) -> set[str]:
    """``spacepresso.X...`` imports, as layer names."""
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "spacepresso"
        ):
            parts = (node.module or "").split(".")
            if len(parts) >= 2:
                out.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("spacepresso."):
                    out.add(alias.name.split(".")[1])
    return out


@pytest.mark.parametrize("module", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_module_only_imports_from_layers_below_it(module: Path):
    layer = _layer_of(module)
    allowed = ALLOWED[layer] | {layer}
    violations = _imports(module) - allowed
    assert not violations, (
        f"{module.relative_to(SRC)} (layer {layer!r}) imports {sorted(violations)}, "
        f"which sit at or above it. Allowed: {sorted(allowed)}."
    )


def test_core_imports_nothing_from_the_project():
    """core is the foundation. If it grows a dependency, the graph has a cycle
    waiting to happen."""
    for module in _modules():
        if _layer_of(module) != "core":
            continue
        assert _imports(module) <= {"core"}, f"{module.name} reaches outside core"


def test_no_detector_imports_another_detector():
    """Two detectors may share machinery, but only through a shared module."""
    shared = {"base", "blocks", "coreset", "training", "prompts", "__init__"}
    for module in _modules():
        if _layer_of(module) != "detectors" or module.stem in shared:
            continue
        tree = ast.parse(module.read_text())
        siblings = {
            (node.module or "").split(".")[2]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("spacepresso.detectors.")
        }
        assert siblings <= shared, (
            f"{module.name} imports sibling detector(s) {sorted(siblings - shared)}. "
            f"Shared machinery belongs in one of {sorted(shared)}."
        )


def test_there_are_no_sys_path_hacks():
    offenders = [
        m.relative_to(SRC) for m in _modules() if "sys.path.insert" in m.read_text()
    ]
    assert not offenders, f"sys.path manipulation in {offenders}"


def test_no_module_level_imports_are_deferred_to_break_cycles():
    offenders: list[str] = []
    for module in _modules():
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.ImportFrom) and (
                    inner.module or ""
                ).startswith("spacepresso"):
                    # The CLI defers imports deliberately, to keep --help fast
                    # and to avoid importing every detector's dependencies.
                    if module.stem == "cli":
                        continue
                    offenders.append(
                        f"{module.relative_to(SRC)}::{node.name} -> {inner.module}"
                    )
    assert not offenders, "deferred project imports (cycle smell): " + ", ".join(
        offenders
    )


def test_no_hardcoded_absolute_paths():
    import re

    # String literals only; docstrings and comments are not paths.
    pattern = re.compile(r"^/(work|workspace|home|mnt|Users)/")
    offenders: list[str] = []
    for module in _modules():
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and pattern.match(node.value)
            ):
                offenders.append(
                    f"{module.relative_to(SRC)}:{node.lineno} {node.value!r}"
                )
    assert not offenders, "hardcoded absolute paths: " + ", ".join(offenders)


def test_library_code_does_not_print():
    """The CLI prints its results, which is what a CLI is for."""
    offenders: list[str] = []
    for module in _modules():
        if module.stem in ("cli",):
            continue
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{module.relative_to(SRC)}:{node.lineno}")
    assert not offenders, "print() in library code: " + ", ".join(offenders)
