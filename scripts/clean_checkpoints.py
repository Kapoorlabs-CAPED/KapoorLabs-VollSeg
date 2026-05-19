"""Trim a directory of Lightning checkpoints down to a chosen subset.

Keeps ``keep_n_first`` earliest, ``keep_n_middle`` centred, and
``keep_n_last`` most-recent epochs; deletes the rest. Filenames are
assumed to contain ``epoch=<N>`` (no step token required).
"""

import os
import re

# Configuration
directory = "/lustre/fsn1/projects/rech/jsy/uzj81mi/models_unet_pytorch/"
keep_n_first = 0
keep_n_middle = 0
keep_n_last = 1
dry_run = False


def parse_checkpoint_name(filename):
    """Return the epoch number from a checkpoint filename, or None.

    Matches ``epoch=<N>`` anywhere in the name (e.g.
    ``xenopus_unet-epoch=199.ckpt``).
    """
    match = re.search(r"epoch=(\d+)", filename)
    return int(match.group(1)) if match else None


def clean_ckpt_files(
    directory, keep_n_first, keep_n_middle, keep_n_last, dry_run=False
):
    files = [f for f in os.listdir(directory) if f.endswith(".ckpt")]

    file_data = []
    for f in files:
        epoch = parse_checkpoint_name(f)
        if epoch is not None:
            file_data.append((f, epoch))

    # Sort by epoch ascending.
    file_data.sort(key=lambda x: x[1])
    sorted_files = [os.path.join(directory, f[0]) for f in file_data]

    n_total = len(sorted_files)
    n_keep = keep_n_first + keep_n_middle + keep_n_last
    if n_total <= n_keep:
        print(f"Nothing to delete. Only {n_total} checkpoint(s) found.")
        return

    first_files = sorted_files[:keep_n_first] if keep_n_first > 0 else []
    last_files = sorted_files[-keep_n_last:] if keep_n_last > 0 else []
    if keep_n_middle > 0:
        middle_start = max(0, (n_total - keep_n_middle) // 2)
        middle_files = sorted_files[middle_start : middle_start + keep_n_middle]
    else:
        middle_files = []

    files_to_keep = set(first_files + middle_files + last_files)
    files_to_delete = [f for f in sorted_files if f not in files_to_keep]

    action = "Would delete" if dry_run else "Deleting"
    for file in files_to_delete:
        print(f"{action}: {os.path.basename(file)}")
        if not dry_run:
            os.remove(file)

    verb = "Would keep" if dry_run else "Kept"
    print(f"\n{verb} {len(files_to_keep)} file(s):")
    print(f"  First {keep_n_first}:  {[os.path.basename(f) for f in first_files]}")
    print(f"  Middle {keep_n_middle}: {[os.path.basename(f) for f in middle_files]}")
    print(f"  Last {keep_n_last}:   {[os.path.basename(f) for f in last_files]}")


if __name__ == "__main__":
    clean_ckpt_files(
        directory, keep_n_first, keep_n_middle, keep_n_last, dry_run=dry_run
    )
