"""Bump src/vollseg/_version.py from the most recent git tag.

Usage::

    python update_version.py        # reads `git describe --tags --abbrev=0`

The tag may optionally start with ``v`` (e.g. ``v33.0.0``) — that prefix
is stripped before the version string is written. If the repo has no
tags yet (e.g. fresh clone before the first release), ``_version.py``
is left untouched so the pre-commit hook doesn't block development.
"""

import subprocess
import sys


def update_version_file() -> None:
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        print(
            "update_version.py: no git tags found; leaving _version.py unchanged",
            file=sys.stderr,
        )
        return

    if tag.startswith("v"):
        tag = tag[1:]
    try:
        parts = tuple(map(int, tag.split(".")))
    except ValueError:
        print(
            f"update_version.py: tag '{tag}' isn't a numeric version; skipping",
            file=sys.stderr,
        )
        return

    with open("src/vollseg/_version.py", "w") as f:
        f.write(f'__version__ = version = "{tag}"\n')
        f.write(f"__version_tuple__ = version_tuple = {parts}\n")


if __name__ == "__main__":
    update_version_file()
