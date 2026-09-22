"""The command-line interface.

Detector flags are **derived from the config dataclass**, so adding a
hyperparameter means adding a field — not adding a field, an argument, and a
line in the constructor call, and remembering to add it to the run-id hash.

    spacepresso run patchcore --backbone dinov2_vits14_reg --input-size 392
    spacepresso run --config configs/detectors/patchcore-dinov2.yaml
    spacepresso stack --runs baseline_out/runs/a baseline_out/runs/b
    spacepresso list
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from spacepresso.core.logging import get_logger, setup_logging

__all__ = ["main"]

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass → argparse
# ─────────────────────────────────────────────────────────────────────────────
def _unwrap_optional(annotation: Any) -> Any:
    """``X | None`` → ``X``; anything else unchanged."""
    if get_origin(annotation) in (None, type(None)):
        return annotation
    args = [a for a in get_args(annotation) if a is not type(None)]
    return args[0] if len(args) == 1 else annotation


def _annotations(config_class: type) -> dict[str, Any]:
    """Resolved field types.

    Every config module uses ``from __future__ import annotations``, so
    ``dataclasses.fields()`` hands back the annotation as a *string*.
    Building argparse from those silently types every list field as a bare
    string, and `--feature-layers 2 3` then fails with "unrecognized
    arguments: 3". ``get_type_hints`` evaluates them for real.
    """
    hints = get_type_hints(config_class)
    return {f.name: hints.get(f.name, f.type) for f in dataclasses.fields(config_class)}


def _add_field(
    parser: argparse.ArgumentParser, field: dataclasses.Field, annotation: Any
) -> None:
    flag = "--" + field.name.replace("_", "-")
    annotation = _unwrap_optional(annotation)
    help_text = (field.metadata or {}).get("help", "")

    if annotation is bool:
        # Both spellings, so a config file default of True stays overridable.
        parser.add_argument(
            flag, dest=field.name, action="store_true", default=None, help=help_text
        )
        parser.add_argument(
            "--no-" + field.name.replace("_", "-"),
            dest=field.name,
            action="store_false",
            help=argparse.SUPPRESS,
        )
        return

    origin = get_origin(annotation)
    if origin in (tuple, list):
        inner = get_args(annotation)
        item_type = _scalar_type(inner[0]) if inner else str
        parser.add_argument(
            flag,
            dest=field.name,
            nargs="*",
            type=item_type,
            default=None,
            help=help_text,
        )
        return

    parser.add_argument(
        flag,
        dest=field.name,
        type=_scalar_type(annotation),
        default=None,
        help=help_text,
    )


def _scalar_type(annotation: Any):
    for candidate in (int, float, Path):
        if annotation is candidate:
            return candidate
    return str


def _apply_overrides(config_class: type, values: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields the dataclass declares, dropping unset flags."""
    annotations = _annotations(config_class)
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key not in annotations or value is None:
            continue
        if get_origin(_unwrap_optional(annotations[key])) is tuple and isinstance(
            value, list
        ):
            value = tuple(value)
        out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Subcommands
# ─────────────────────────────────────────────────────────────────────────────
def _cmd_list(_args: argparse.Namespace) -> int:
    import sys as _sys

    from spacepresso.detectors import get_detector, list_detectors

    print("Detectors:")
    for name in list_detectors():
        detector, _config = get_detector(name)
        # The class docstring when there is one, otherwise the module's first
        # line — which is where each detector's one-line summary lives.
        module = _sys.modules[detector.__module__]
        for source in (detector.__doc__, module.__doc__):
            lines = (source or "").strip().splitlines()
            if lines:
                print(f"  {name:22s} {lines[0]}")
                break
        else:
            print(f"  {name}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from spacepresso.config import RuntimeConfig
    from spacepresso.detectors import get_detector
    from spacepresso.runner.experiment import run_experiment

    file_values: dict[str, Any] = {}
    if args.config:
        file_values = _load_yaml(Path(args.config))

    name = args.detector or file_values.pop("detector", None)
    if not name:
        raise SystemExit(
            "no detector given. Pass one positionally, or set 'detector:' in "
            "the config file."
        )

    detector_class, config_class = get_detector(name)
    cli_values = vars(args)

    detector_values = {
        **file_values.get("config", {}),
        **_apply_overrides(config_class, cli_values),
    }
    runtime_values = {
        **file_values.get("runtime", {}),
        **_apply_overrides(RuntimeConfig, cli_values),
    }

    config = config_class(**detector_values)
    runtime = RuntimeConfig(**runtime_values)

    result = run_experiment(lambda c, r: detector_class(c, r), config, runtime)
    print(f"\nrun_id: {result.run_id}")
    print(f"run_dir: {result.run_dir}")
    if result.submission_path:
        print(f"submission: {result.submission_path}")
    return 0


def _cmd_stack(args: argparse.Namespace) -> int:
    from spacepresso.stacking import StackerConfig, run_stacker
    from spacepresso.stacking.features import FeatureConfig

    file_values: dict[str, Any] = {}
    if args.config:
        file_values = _load_yaml(Path(args.config))

    values: dict[str, Any] = dict(file_values)
    feature_values = values.pop("features", {})
    values.update(_apply_overrides(StackerConfig, vars(args)))

    if args.runs:
        values["runs"] = tuple(Path(r) for r in args.runs)
    elif "runs" in values:
        values["runs"] = tuple(Path(r) for r in values["runs"])

    values["features"] = FeatureConfig(**feature_values)
    result = run_stacker(StackerConfig(**values))

    print(f"\nrun_id: {result.run_id}")
    print(f"overall out-of-fold pooled AP: {result.overall_ap:.4f}")
    if result.submission_path:
        print(f"submission: {result.submission_path}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    from spacepresso.config import RuntimeConfig
    from spacepresso.detectors import get_detector, list_detectors
    from spacepresso.stacking.stacker import StackerConfig

    parser = argparse.ArgumentParser(
        prog="spacepresso",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="show the registered detectors").set_defaults(
        func=_cmd_list
    )

    run = subparsers.add_parser(
        "run",
        help="run one detector over the dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run.add_argument(
        "detector", nargs="?", choices=list_detectors(), help="which detector to run"
    )
    run.add_argument("--config", type=Path, help="YAML config file")
    run.set_defaults(func=_cmd_run)

    runtime_group = run.add_argument_group("runtime (does not affect results)")
    runtime_hints = _annotations(RuntimeConfig)
    for field in dataclasses.fields(RuntimeConfig):
        _add_field(runtime_group, field, runtime_hints[field.name])

    # Every detector's fields, unioned. Flags shared between detectors (
    # --backbone, --input-size) are added once; a flag only one detector has
    # is simply ignored by the others, and the config constructor rejects it
    # if it is passed to a detector that does not declare it.
    detector_group = run.add_argument_group("detector settings")
    seen: set[str] = set(runtime_hints)
    for name in list_detectors():
        _detector, config_class = get_detector(name)
        hints = _annotations(config_class)
        for field in dataclasses.fields(config_class):
            if field.name in seen or field.name.startswith("_"):
                continue
            seen.add(field.name)
            _add_field(detector_group, field, hints[field.name])

    stack = subparsers.add_parser(
        "stack",
        help="combine several detector runs into one submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stack.add_argument("--config", type=Path, help="YAML config file")
    stack.add_argument("--runs", nargs="*", type=Path, help="run directories to stack")
    stack.set_defaults(func=_cmd_stack)
    stack_hints = _annotations(StackerConfig)
    for field in dataclasses.fields(StackerConfig):
        if field.name in ("runs", "features"):
            continue
        _add_field(stack, field, stack_hints[field.name])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
