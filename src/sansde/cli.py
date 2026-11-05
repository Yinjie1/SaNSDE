"""Command line interface for the sansde package.

Usage:
    sansde run params.json            # run in the background, log to params.log
    sansde run params.json --fg       # run in the foreground
    sansde xyz2vasp params.json       # export VASP tasks from JSON parameters
    python -m sansde run params.json --fg
"""
import argparse
import subprocess
import sys
from pathlib import Path

from sansde.io import write_vasp_tasks
from sansde.main import main as run_opt


def main(argv: list[str] | None = None) -> None:
    """Parse command line arguments and dispatch the requested task."""
    parser = argparse.ArgumentParser(prog="sansde")
    subparsers = parser.add_subparsers(dest="command")

    export = subparsers.add_parser(
        "xyz2vasp",
        help="export xyz directories to VASP task directories using JSON parameters",
    )
    export.add_argument(
        "json",
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

    if args.command == "xyz2vasp":
        write_vasp_tasks(args.json)
        return

    if args.command != "run":
        parser.error("missing command")

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
