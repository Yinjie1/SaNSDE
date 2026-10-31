"""Command line entry point for the sansde package.

Usage:
    sansde --opt params.json            # 默认后台运行, 日志写到 params.log
    sansde --opt params.json --fg       # 前台运行
    sansde xyz2vasp -i gen1 gen2 --cell 15 15 15 -c -o tasks
    sansde xyz2vasp --frame 3 -s 0.99 0.98
    python -m sansde --opt params.json --fg
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from ase.io import write as ase_write

from sansde.io import merge_xyz_directories, write_vasp_tasks
from sansde.main import run


def main(argv: list[str] | None = None) -> None:
    """Parse command line arguments and dispatch the requested task.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list. ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(prog="sansde")
    subparsers = parser.add_subparsers(dest="command")

    export = subparsers.add_parser(
        "xyz2vasp",
        help="export xyz directories to VASP task directories",
    )
    export.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=Path,
        default=[Path("best")],
        metavar="XYZ_DIR",
        help="directories containing xyz files (default: best)",
    )
    export.add_argument(
        "--frame",
        "--flame",
        dest="frame",
        type=int,
        default=3,
        metavar="N",
        help="frames read from each xyz file (default: 3)",
    )
    export.add_argument(
        "-s",
        "--similarity",
        nargs=2,
        type=float,
        default=(0.99, 0.98),
        metavar=("S", "S_C1"),
        help="similarity thresholds for non-C1 and C1 point groups "
        "(default: 0.99 0.98)",
    )
    export.add_argument(
        "--cell",
        nargs=3,
        type=float,
        metavar=("A", "B", "C"),
        help="orthorhombic cell lengths in Angstrom; overrides cluster cells",
    )
    export.add_argument(
        "-c",
        "--compress",
        action="store_true",
        help="pack the output into a .tgz file and remove the directory",
    )
    export.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="OUT",
        help="output directory name (default: relax.<elements>)",
    )
    export.add_argument(
        "-t",
        "--template",
        type=Path,
        default=Path("input"),
        metavar="DIR",
        help="VASP input template directory (default: input)",
    )

    parser.add_argument(
        "--opt",
        type=Path,
        metavar="PARAMS",
        help="run the optimization with the given params JSON file",
    )
    parser.add_argument(
        "--fg",
        "--foreground",
        dest="foreground",
        action="store_true",
        help="run in the foreground (default: run in the background)",
    )
    args = parser.parse_args(argv)

    if args.command == "xyz2vasp":
        s, s_c1 = args.similarity
        clusters = merge_xyz_directories(
            args.input,
            n=args.frame,
            s=s,
            s_c1=s_c1,
        )
        print(f"loaded {len(clusters)} clusters from {len(args.input)} input path(s)")
        elements = sorted(
            {symbol for cluster in clusters for symbol in cluster.get_chemical_symbols()}
        )
        output = (
            args.output
            if args.output is not None
            else Path(f"relax.{''.join(elements)}")
        )
        cell = args.cell if args.cell is not None else [None, None, None]

        work_dir = Path.cwd()
        if clusters:
            ase_write(work_dir / "merged.xyz", list(clusters), format="extxyz")
        xyz_dir = work_dir / "merged_xyz"
        json_dir = work_dir / "merged_json"
        xyz_dir.mkdir(exist_ok=True)
        json_dir.mkdir(exist_ok=True)
        counts: dict[str, int] = {}
        for cluster in clusters:
            formula = cluster.get_chemical_formula()
            index = counts.get(formula, 0)
            counts[formula] = index + 1
            ase_write(xyz_dir / f"{formula}.{index}.xyz", cluster, format="extxyz")
            record = {
                "symbols": cluster.get_chemical_symbols(),
                "positions": cluster.get_positions().tolist(),
                "cell": cluster.cell.array.tolist(),
                "pbc": cluster.pbc.tolist(),
                "info": {
                    key: value.tolist() if hasattr(value, "tolist") else value
                    for key, value in cluster.info.items()
                },
            }
            (json_dir / f"{formula}.{index}.json").write_text(
                json.dumps(record, indent=2)
            )
        print(
            f"wrote merged.xyz, merged_xyz/ and merged_json/ "
            f"({len(clusters)} frames)"
        )

        tasks = write_vasp_tasks(
            clusters,
            output,
            args.template,
            *cell,
            pack=args.compress,
        )
        print(f"wrote {len(tasks)} VASP tasks to {output}")
        return

    if args.opt is None:
        parser.error("missing --opt PARAMS")

    if args.foreground:
        # 显式前台: 直接在当前进程里跑, 输出留在终端.
        run(args.opt)
        return

    # 默认后台: 重新拉起一个脱离当前会话的子进程.
    # 子进程显式带 --fg, 避免再次进入后台分支.
    log_path = args.opt.with_suffix(".log")
    command = [
        sys.executable,
        "-u",  # 日志不缓冲, 便于 tail 实时查看
        "-m",
        "sansde",
        "--opt",
        str(args.opt),
        "--fg",
    ]
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,  # 不依赖终端输入
            stdout=log_file,
            stderr=subprocess.STDOUT,  # 标准错误也写进同一个日志
            # start_new_session=True 相当于 setsid, 终端退出后子进程继续运行
            start_new_session=True,
        )
    # 记录后台任务 pid, 每行一个, 便于后续批量管理.
    pid_path = Path("pid")
    with pid_path.open("a", encoding="utf-8") as pid_file:
        pid_file.write(f"{process.pid}\n")
    print(f"started background process pid={process.pid}, log={log_path}")
    return


if __name__ == "__main__":
    main()
