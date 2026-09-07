"""Manual entry points; importing or planning never submits a calculation."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="Print configuration only; no Torch/CUDA import"
    )
    plan.add_argument("figure", choices=["fig2", "fig3", "fig4", "fig5", "fig6"])
    validate = commands.add_parser(
        "validate", help="Check geometry and required input checksums, CPU only"
    )
    validate.add_argument("--inputs", type=Path, default=Path("inputs"))
    run = commands.add_parser(
        "run", help="Run a figure now, explicitly and in the foreground"
    )
    run.add_argument("figure", choices=["fig2", "fig3", "fig4", "fig5", "fig6"])
    run.add_argument("--output", type=Path)
    run.add_argument("--inputs", type=Path, default=Path("inputs"))
    run.add_argument("--device", default="cuda")
    run.add_argument(
        "--sources",
        type=int,
        nargs="+",
        help="Fig6 source subset for validation; omitted means all sources",
    )
    run.add_argument("--evolution-batch", type=int, help="Fig6 norm-BP/source-wave cap")
    run.add_argument(
        "--read-batch", type=int, help="Fig6 maximum adaptive readout batch"
    )
    plot = commands.add_parser(
        "plot", help="Render only completed matching results; no calculation"
    )
    plot.add_argument("figure", choices=["fig2", "fig3", "fig4", "fig5", "fig6"])
    plot.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        from residual_tn.config import describe

        print(json.dumps(describe(args.figure), indent=2))
    elif args.command == "validate":
        from residual_tn.geometry import validate_geometry
        from residual_tn.paths import CONFIGS
        from residual_tn.identity import sha256

        manifest = json.loads((CONFIGS / "residual_manifest.json").read_text())
        audits = [validate_geometry(g) for g in manifest["geometries"]]
        hashes = json.loads((args.inputs / "sha256.json").read_text())
        for name, row in hashes.items():
            if sha256(args.inputs / name) != row["sha256"]:
                raise ValueError("Input checksum mismatch: " + name)
        print(
            json.dumps(
                dict(passed=True, geometries=audits, inputs=list(hashes)), indent=2
            )
        )
    elif args.command == "run":
        from residual_tn.config import read

        output = args.output or Path("results") / args.figure
        if args.figure == "fig6":
            from residual_tn.experiments.pauli_peps import run

            run(
                read(args.figure),
                output,
                args.device,
                args.sources,
                args.evolution_batch,
                args.read_batch,
            )
        else:
            if (
                args.sources is not None
                or args.evolution_batch is not None
                or args.read_batch is not None
            ):
                parser.error(
                    "Source/batch overrides are available only for Fig6; Fig2–5 use their fixed study configurations"
                )
            from residual_tn.studies import run_study

            run_study(args.figure, output, args.inputs, args.device)
    else:
        render(args.figure, args.results)


def render(figure, directory):
    directory = Path(directory)
    if figure == "fig6":
        from residual_tn.figures.fig6 import render

        render(directory)
    elif figure == "fig5":
        from residual_tn.figures.fig5 import render

        render(directory)
    else:
        from residual_tn.figures import fig234 as builder

        builder.DATA = directory / "data"
        builder.OUT = directory / "figures"
        builder.OUT.mkdir(parents=True, exist_ok=True)
        if figure == "fig4":
            builder.plot_fig4(
                mc_json_path=directory / "mc/qutrit_mc_4x4_vs_fullstate_f2.json",
                samples_path=directory / "mc/qutrit_mc_4x4_samples.npz",
                pauli_json_path=directory / "pauli/pauli_16q_fullstate.json",
                output_dir=builder.OUT,
            )
        else:
            getattr(builder, "plot_" + figure)()
