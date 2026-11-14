"""Command line interface for the sansde package.

Usage:
    sansde init                       # write params.json with default values
    sansde run params.json            # run in the background, log to params.log
    sansde run params.json --fg       # run in the foreground
    sansde xyz2vasp params.json       # export VASP tasks from JSON parameters
    sansde check [directory]          # inspect run parameter files and outputs
    sansde analyse -i json_dir        # analyse ASE JSON cluster structures
    sansde radial -i json_dir         # analyse radial layer structure
    sansde cn -i json_dir             # analyse coordination numbers
    python -m sansde run params.json --fg
"""
import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from sansde.analysis import (
    analyze_clusters,
    analyze_coordination,
    analyze_radial,
    write_cluster_csv,
    write_coordination_csv,
    write_radial_csv,
)
from sansde.check import run_check
from sansde.io import DEFAULT_RUN_PARAMS, load_clusters, write_vasp_tasks
from sansde.main import main as run_opt
from sansde.plots import (
    plot_binding,
    plot_coordination,
    plot_energy,
    plot_radial,
    plot_reference,
    plot_shape,
    plot_similarity,
)

_RUN_DESCRIPTIONS: dict[str, str] = {
    "EXPORT": "Environment variables applied before the numerical libraries are imported and before SCRIPTS.",
    "SCRIPTS": "Python snippets executed in order before POTENTIAL is evaluated.",
    "ELEMENT": "Element symbol of the cluster.",
    "ELEMENT2": "Second element symbol. Non-empty enables fixed-composition alloy clusters.",
    "TOTAL_ATOM": "Total atom count of an alloy cluster. Must be larger than MAX_ATOM.",
    "LATTICE_CONST": "Initial bond-length scale for random clusters, in Angstrom.",
    "CELL": "Cell [a, b, c, alpha, beta, gamma] attached while a calculator runs. null disables it.",
    "PBC": "Periodic boundary flags used with CELL. A single boolean applies to all axes. Ignored without CELL.",
    "MIN_ATOM": "Smallest ELEMENT atom count to optimize, inclusive.",
    "MAX_ATOM": "Largest ELEMENT atom count to optimize, inclusive.",
    "ROUND": "Number of optimization rounds. Later rounds start from the previous best.",
    "INIT_DIR": "Directory with {element}{atom}*.xyz initial structures. Empty disables loading.",
    "INIT_POP_DIR": "Directory with population snapshots. Takes precedence over INIT_DIR.",
    "BEST_DIR": "Output directory for best structures.",
    "BETTER_DIR": "Output directory for structures that improved the search.",
    "KEEP_SNAPSHOTS": "Number of population snapshots kept per size.",
    "POP_DIR": "Output directory for population snapshots.",
    "POP_SIZE": "Population size.",
    "GROW_PREV": "Initial structures grown from the previous-size best by adding one atom. 0 disables it; must be < POP_SIZE - 1.",
    "MAX_GEN": "Maximum number of generations.",
    "CONV_STEPS": "Stagnant generations before stopping. 0 disables the early stop.",
    "EDIFF": "Energy tolerance for stagnation.",
    "SELECT_TOLE": "Energy tolerance for selection. null falls back to EDIFF.",
    "S_STRICT": "Similarity threshold for a clear structure change.",
    "S_LOOSE": "Similarity threshold used together with EDIFF.",
    "CAND_SIZE": "Number of best structures kept per size.",
    "FP_ADAPTIVE": "Adapt fp together with p. false keeps fp at 0.5.",
    "LEARNING_PERIOD": "Generations between p and fp updates. 0 disables the update.",
    "CR_PERIOD": "Generations between CR resampling. 0 disables resampling.",
    "CRM_PERIOD": "Generations between CRm updates. 0 disables the update.",
    "CONTINUE": "Import the whole population from INIT_POP_DIR. false imports only the best structure and resets the adaptive state.",
    "POTENTIAL": "ASE calculator expression evaluated after SCRIPTS.",
    "USE_ZBL": "Add the built-in ZBL short-range repulsion on top of POTENTIAL.",
    "ZBL_CUTOFF": "ZBL cutoff in Angstrom.",
    "ZBL_SMOOTH": "Smooth switch at the ZBL cutoff.",
    "RELAX_FMAX": "BFGS force tolerance for local relaxation, in eV/Angstrom.",
    "RELAX_STEPS": "Maximum BFGS steps for local relaxation.",
}

_RUN_TYPES: dict[str, str] = {
    "EXPORT": "dict",
    "SCRIPTS": "list[str]",
    "CELL": "6 numbers or null",
    "PBC": "bool, 3 booleans or null",
    "SELECT_TOLE": "float or null",
}

_EXPORT_PARAMETERS: tuple[tuple[str, str, str, str], ...] = (
    ("input", "str or list[str]", '"best"', "xyz files or directories to merge."),
    ("template", "str", '"input"', "VASP template directory with INCAR, KPOINTS and optionally POTCAR."),
    ("frame", "int", "3", "Number of frames read from the end of each xyz file."),
    ("similarity", "list[float]", "[0.99, 0.98]", "Deduplication thresholds [s, s_c1]."),
    ("cell", "6 numbers or null", "null", "Cell [a, b, c, alpha, beta, gamma] for small clusters."),
    ("large_cluster_cell", "6 numbers", "[20.0, 20.0, 20.0, 90.0, 90.0, 90.0]", "Cell for clusters with at least large_cluster_atom_count atoms."),
    ("large_cluster_atom_count", "int", "40", "Atom-count threshold for large_cluster_cell and large_cluster_unique."),
    ("large_cluster_unique", "bool", "true", "Keep only the lowest-energy structure per formula for large clusters."),
    ("output", "str or null", "null", "Output directory name. null means relax.<elements>."),
    ("compress", "bool", "false", "Pack the output directory into a tgz archive."),
)


def _run_parameter_type(name: str, default: object) -> str:
    """Return the JSON type name of one run parameter."""
    try:
        return _RUN_TYPES[name]
    except KeyError:
        pass
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, dict):
        return "dict"
    if isinstance(default, list):
        return "list"
    return "str"


def _print_run_manual() -> None:
    """Print the params.json parameter manual."""
    print("params.json parameter manual")
    print()
    for name, default in DEFAULT_RUN_PARAMS.items():
        print(f"{name} ({_run_parameter_type(name, default)}, default {json.dumps(default)}):")
        print(
            textwrap.fill(
                _RUN_DESCRIPTIONS.get(name, ""),
                width=88,
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )
        print()


def _print_export_manual() -> None:
    """Print the xyz2vasp parameter manual."""
    print("xyz2vasp parameter manual")
    print()
    for name, kind, default, description in _EXPORT_PARAMETERS:
        print(f"{name} ({kind}, default {default}):")
        print(
            textwrap.fill(
                description,
                width=88,
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )
        print()

def main(argv: list[str] | None = None) -> None:
    """Parse command line arguments and dispatch the requested task."""
    parser = argparse.ArgumentParser(prog="sansde")
    subparsers = parser.add_subparsers(dest="command")

    initializer = subparsers.add_parser(
        "init",
        help="write params.json with default values",
        add_help=False,
    )
    initializer.add_argument(
        "-h",
        "--help",
        dest="manual",
        action="store_true",
        help="show the params.json parameter manual",
    )

    analyser = subparsers.add_parser(
        "analyse",
        help="run cluster analysis on ASE JSON structures",
    )
    analyser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="JSON_DIR",
        help="directory containing ASE JSON structures",
    )
    analyser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="CSV",
        help="CSV output path (default: standard output)",
    )
    analyser.add_argument(
        "--s2",
        action="store_true",
        help="also compute the angle similarity S2",
    )
    analyser.add_argument(
        "--ref",
        type=Path,
        default=None,
        metavar="PATH",
        help="reference ASE JSON directory or CSV file",
    )
    analyser.add_argument(
        "--e1",
        type=float,
        default=None,
        metavar="E1",
        help="isolated atom energy; defaults to the N=1 structure",
    )
    analyser.add_argument(
        "--plot",
        nargs="*",
        choices=("svg", "png", "eps"),
        default=None,
        metavar="FORMAT",
        help="write PNG plots by default; pass one or more of svg, png and eps",
    )
    analyser.add_argument(
        "--shape-start",
        type=float,
        default=6.0,
        metavar="N",
        help="first atom count shown in the shape plot (default: 6)",
    )

    radial = subparsers.add_parser(
        "radial",
        help="analyse radial layer structure from ASE JSON structures",
    )
    radial.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="JSON_DIR",
        help="directory containing ASE JSON structures",
    )
    radial.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="CSV",
        help="CSV output path (default: standard output)",
    )
    radial.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=None,
        metavar="K",
        help="manual layer counts in ascending atom-count order",
    )
    radial.add_argument(
        "--plot",
        nargs="*",
        choices=("svg", "png", "eps"),
        default=None,
        metavar="FORMAT",
        help="write PNG plots by default; pass one or more of svg, png and eps",
    )
    radial.add_argument(
        "--cyan-pos",
        type=float,
        default=1.0 / 6.0,
        metavar="POS",
        help="cyan position in the overlap color map (default: 1/6)",
    )

    coordination = subparsers.add_parser(
        "cn",
        help="analyse coordination numbers from ASE JSON structures",
    )
    coordination.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="JSON_DIR",
        help="directory containing ASE JSON structures",
    )
    coordination.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="CSV",
        help="CSV output path (default: standard output)",
    )
    coordination.add_argument(
        "--cutoff",
        type=float,
        default=None,
        metavar="A",
        help="bonding cutoff; defaults to the bulk neighbor mean distance",
    )
    coordination.add_argument(
        "--bulk-cn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compute and plot the bulk average coordination number",
    )
    coordination.add_argument(
        "--plot",
        nargs="*",
        choices=("svg", "png", "eps"),
        default=None,
        metavar="FORMAT",
        help="write PNG plots by default; pass one or more of svg, png and eps",
    )

    checker = subparsers.add_parser(
        "check",
        help="check run parameter files and output directories in a directory",
    )
    checker.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("."),
        help="directory containing parameter JSON files (default: current directory)",
    )
    checker.add_argument(
        "--no-banner",
        action="store_true",
        help="do not print the SANSDE banner",
    )

    export = subparsers.add_parser(
        "xyz2vasp",
        help="export xyz directories to VASP task directories using JSON parameters",
        add_help=False,
    )
    export.add_argument(
        "-h",
        "--help",
        dest="manual",
        action="store_true",
        help="show the xyz2vasp parameter manual",
    )
    export.add_argument(
        "json",
        nargs="?",
        type=Path,
        metavar="PARAMS",
        help="JSON parameter file, e.g. sansde xyz2vasp params.json",
    )

    runner = subparsers.add_parser(
        "run",
        help="run the optimization with a params JSON file",
    )
    runner.add_argument(
        "params",
        type=Path,
        metavar="PARAMS",
        help="JSON parameter file, e.g. sansde run params.json",
    )
    runner.add_argument(
        "--fg",
        "--foreground",
        dest="foreground",
        action="store_true",
        help="run in the foreground (default: run in the background)",
    )
    args = parser.parse_args(argv)

    if args.command == "init":
        if args.manual:
            _print_run_manual()
            return
        output = Path("params.json")
        output.write_text(
            json.dumps(DEFAULT_RUN_PARAMS, indent=4) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output}")
        return

    if args.command == "analyse":
        clusters = load_clusters(args.input)
        rows = analyze_clusters(
            clusters,
            e1=args.e1,
            ref=args.ref,
            compute_s2=args.s2,
        )
        write_cluster_csv(rows, args.output)
        if args.plot is not None:
            input_path = args.input.resolve()
            prefix = input_path.stem if input_path.is_file() else input_path.name
            formats = list(dict.fromkeys(args.plot or ["png"]))
            for suffix in formats:
                plot_energy(rows, f"{prefix}_energy.{suffix}")
                plot_shape(
                    rows,
                    f"{prefix}_shape.{suffix}",
                    shape_start=args.shape_start,
                )
                plot_similarity(rows, f"{prefix}_s.{suffix}")
                plot_reference(rows, f"{prefix}_energy_ref.{suffix}")
                plot_binding(rows, f"{prefix}_eb.{suffix}")
        return

    if args.command == "cn":
        clusters = load_clusters(args.input)
        rows = analyze_coordination(
            clusters,
            cutoff=args.cutoff,
            show_bulk_cn=args.bulk_cn,
        )
        write_coordination_csv(rows, args.output)
        if args.plot is not None:
            input_path = args.input.resolve()
            prefix = input_path.stem if input_path.is_file() else input_path.name
            formats = list(dict.fromkeys(args.plot or ["png"]))
            for suffix in formats:
                plot_coordination(rows, f"{prefix}_cn.{suffix}")
        return

    if args.command == "radial":
        clusters = load_clusters(args.input)
        rows = analyze_radial(clusters, k=args.k)
        write_radial_csv(rows, args.output)
        if args.plot is not None:
            input_path = args.input.resolve()
            prefix = input_path.stem if input_path.is_file() else input_path.name
            formats = list(dict.fromkeys(args.plot or ["png"]))
            for suffix in formats:
                plot_radial(
                    rows,
                    f"{prefix}_radial.{suffix}",
                    cyan_pos=args.cyan_pos,
                )
        return

    if args.command == "check":
        run_check(args.directory, show_banner=not args.no_banner)
        return

    if args.command == "xyz2vasp":
        if args.manual:
            _print_export_manual()
            return
        if args.json is None:
            parser.error("xyz2vasp requires a JSON parameter file")
        write_vasp_tasks(args.json)
        return

    if args.command != "run":
        parser.error("missing command")

    if not Path(args.params).is_file():
        print(f"error: parameter file not found: {args.params}", file=sys.stderr)
        raise SystemExit(1)

    if args.foreground:
        # Explicit foreground: run in the current process, output stays on the terminal.
        run_opt(args.params)
        return

    # Default background mode: detach a child process from the current session.
    # The child carries --fg so it does not enter the background branch again.
    log_path = args.params.with_suffix(".log")
    command = [
        sys.executable,
        "-u",  # unbuffered log, convenient for live tail
        "-m",
        "sansde",
        "run",
        str(args.params),
        "--fg",
    ]
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,  # do not depend on terminal input
            stdout=log_file,
            stderr=subprocess.STDOUT,  # write stderr to the same log
            # start_new_session=True is equivalent to setsid: the child keeps
            # running after the terminal exits.
            start_new_session=True,
        )
    # Record background pids, one per line, for later batch management.
    pid_path = Path("pid")
    with pid_path.open("a", encoding="utf-8") as pid_file:
        pid_file.write(f"{process.pid}\n")
    print(f"started background process pid={process.pid}, log={log_path}")
