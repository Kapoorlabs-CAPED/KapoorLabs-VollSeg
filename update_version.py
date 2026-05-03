"""Bump src/vollseg/_version.py from the most recent git tag.

Usage::

    python update_version.py        # reads `git describe --tags --abbrev=0`

The tag may optionally start with ``v`` (e.g. ``v33.0.0``) — that prefix
is stripped before the version string is written.
"""

import subprocess


def update_version_file() -> None:
    tag = (
        subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
        .decode()
        .strip()
    )
    if tag.startswith("v"):
        tag = tag[1:]
    parts = tuple(map(int, tag.split(".")))
    with open("src/vollseg/_version.py", "w") as f:
        f.write(f'__version__ = version = "{tag}"\n')
        f.write(f"__version_tuple__ = version_tuple = {parts}\n")


if __name__ == "__main__":
    update_version_file()
