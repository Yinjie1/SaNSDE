"""Command line entry point for the sansde package.

Usage:
    sansde --opt params.json            # 默认后台运行, 日志写到 params.log
    sansde --opt params.json --fg       # 前台运行
    python -m sansde --opt params.json --fg
"""
import argparse
import subprocess
import sys
from pathlib import Path

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
    print(f"started background process pid={process.pid}, log={log_path}")
    return


if __name__ == "__main__":
    main()
