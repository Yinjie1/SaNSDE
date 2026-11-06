"""Miscellaneous shared helpers."""

import shutil
import sys

_BANNER_LINES = (
    "███████╗ █████╗ ███╗   ██╗███████╗██████╗ ███████╗",
    "██╔════╝██╔══██╗████╗  ██║██╔════╝██╔══██╗██╔════╝",
    "███████╗███████║██╔██╗ ██║███████╗██║  ██║█████╗",
    "╚════██║██╔══██║██║╚██╗██║╚════██║██║  ██║██╔══╝",
    "███████║██║  ██║██║ ╚████║███████║██████╔╝███████╗",
    "╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═════╝ ╚══════╝",
)
_BANNER_WIDTH = max(len(line) for line in _BANNER_LINES)


def print_banner() -> None:
    """Print the SANSDE banner, skipping it on a narrow terminal.

    Non-terminal output (redirected or background) always prints the banner.
    """
    if sys.stdout.isatty():
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        if width < _BANNER_WIDTH:
            return
    print("\n".join(_BANNER_LINES))
    print()
