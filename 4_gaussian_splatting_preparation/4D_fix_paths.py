#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4D_fix_paths.py

After unzipping data.zip on a new machine, rewrite the absolute paths
embedded by Nerfstudio inside every splatfacto config.yml so they match
the current project root.

Nerfstudio serializes paths as pathlib.PosixPath objects, distributed
across multiple YAML lines. This script:
  1. Scans every config.yml under
        data/data_for_gaussian_splatting/<BAG>/outputs/splatfacto_split_*/splatfacto/<ts>/
  2. Finds every PosixPath sequence whose first component is "/" (absolute)
  3. Rewrites the leading components up to and including ".../<project_root>/"
     with the components of the current project root
  4. Leaves untouched any PosixPath that is relative (e.g. colmap_path,
     images_path, relative_log_dir)

Idempotent: running it twice does nothing on the second run.

Usage (from project root):
    python 4_gaussian_splatting_preparation/4D_fix_paths.py
    python 4_gaussian_splatting_preparation/4D_fix_paths.py --dry_run
"""

import argparse
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

GS_CONFIGS_GLOB = (
    "data/data_for_gaussian_splatting/*/outputs/splatfacto_split_*/splatfacto/*/config.yml"
)


# Matches a YAML PosixPath sequence with absolute prefix:
#     <indent>key: <maybe-anchor> !!python/object/apply:pathlib.PosixPath
#     <indent>- /
#     <indent>- home
#     <indent>- <user>
#     <indent>- ...
#     <indent>- <leaf>
# We capture:
#   - the header line (the key: ... part)
#   - the indented "- /" line
#   - all subsequent "- <component>" lines (any depth)
#
# The block stops when a line at the same indent doesn't start with "- " or
# when indent decreases.
POSIXPATH_HEADER_RE = re.compile(
    r"^(?P<indent>[ ]*)(?P<key>[A-Za-z_][A-Za-z_0-9]*):\s*"
    r"(?P<anchor>&\S+\s+)?"
    r"!!python/object/apply:pathlib\.PosixPath\s*$"
)


def find_posixpath_blocks(text):
    """
    Yield (start_idx, end_idx, indent, components) for every PosixPath block
    whose first component is "/".

    start_idx / end_idx are byte offsets in text marking the slice that
    contains ONLY the "- <component>" lines (not the header line).
    """
    lines = text.splitlines(keepends=True)
    line_offsets = []
    off = 0
    for ln in lines:
        line_offsets.append(off)
        off += len(ln)

    i = 0
    while i < len(lines):
        m = POSIXPATH_HEADER_RE.match(lines[i].rstrip("\n"))
        if not m:
            i += 1
            continue

        header_indent = m.group("indent")
        # Component lines must be indented at "<header_indent>"
        # and start with "- ".
        comp_prefix = header_indent + "- "

        j = i + 1
        components = []
        while j < len(lines):
            raw = lines[j].rstrip("\n")
            if raw.startswith(comp_prefix):
                # Extract the component value (strip "- " and optional quotes).
                val = raw[len(comp_prefix):].strip()
                # YAML may have wrapped some values in quotes.
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                components.append(val)
                j += 1
            else:
                break

        # Only rewrite ABSOLUTE PosixPath blocks (first component == "/").
        if components and components[0] == "/":
            start_idx = line_offsets[i + 1]  # first "- /" line
            end_idx = line_offsets[j] if j < len(lines) else len(text)
            yield (i + 1, j, header_indent, components, start_idx, end_idx)

        i = j


def rewrite_block(text, header_indent, old_components, new_components):
    """
    Build the replacement text for a PosixPath block.
    Each component becomes "<indent>- <value>\\n".
    """
    # Determine how many leading components belong to the "project root"
    # we want to substitute. We look for the suffix that we want to KEEP:
    # everything from "data" onwards if present, otherwise we replace ALL
    # components (path was project-root itself).
    try:
        # Find "data" component (Nerfstudio paths always contain it because
        # everything lives under data/data_for_gaussian_splatting/...).
        # If you ever change the on-disk layout, edit this anchor.
        data_idx = old_components.index("data")
    except ValueError:
        # No "data" component -> the absolute path IS the project root.
        # Replace the whole component list with new_components.
        kept_tail = []
        full_new = new_components
    else:
        kept_tail = old_components[data_idx:]
        full_new = new_components + kept_tail

    new_lines = []
    for comp in full_new:
        # Re-quote values that yaml would normally quote (e.g. "0", empty).
        # PyYAML uses single quotes for purely numeric or special values.
        if comp == "" or (comp.isdigit()) or comp in (
            "true", "false", "null", "yes", "no", "on", "off"
        ):
            comp_yaml = f"'{comp}'"
        else:
            comp_yaml = comp
        new_lines.append(f"{header_indent}- {comp_yaml}\n")
    return "".join(new_lines), kept_tail


def rewrite_config(config_path: Path, new_project_root: Path, dry_run: bool):
    text = config_path.read_text()
    original = text

    new_components = [p for p in str(new_project_root).split("/") if p != ""]
    # PosixPath also has the leading "/" component:
    new_components_with_root = ["/"] + new_components

    blocks = list(find_posixpath_blocks(text))
    if not blocks:
        return False

    # Apply replacements in reverse so offsets don't shift.
    blocks_sorted = sorted(blocks, key=lambda b: b[4], reverse=True)
    new_text = text
    n_changed_in_file = 0

    for (start_line, end_line, header_indent, old_components,
         start_idx, end_idx) in blocks_sorted:
        replacement, kept_tail = rewrite_block(
            new_text, header_indent, old_components, new_components_with_root
        )
        # Skip if the rewrite is a no-op
        existing_slice = new_text[start_idx:end_idx]
        if existing_slice == replacement:
            continue
        new_text = new_text[:start_idx] + replacement + new_text[end_idx:]
        n_changed_in_file += 1

    if new_text == original:
        print(f"[OK ] {config_path}  (no changes)")
        return False

    if dry_run:
        print(f"[DRY] {config_path}  (would rewrite {n_changed_in_file} block(s))")
    else:
        # Backup the original once
        backup = config_path.with_suffix(".yml.bak")
        if not backup.exists():
            backup.write_text(original)
        config_path.write_text(new_text)
        print(f"[FIX] {config_path}  ({n_changed_in_file} block(s))")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix absolute paths embedded by Nerfstudio in config.yml "
                    "files after unzipping data.zip on a new machine."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would change without writing.",
    )
    parser.add_argument(
        "--project_root", type=str, default=None,
        help="Override the project root used for the rewrite "
             "(default: parent of this script).",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root \
        else PROJECT_ROOT

    print(f"[INFO] Project root: {project_root}")

    configs = sorted(project_root.glob(GS_CONFIGS_GLOB))
    if not configs:
        print(f"[ERROR] No config.yml found under "
              f"{project_root}/{GS_CONFIGS_GLOB}")
        sys.exit(1)

    print(f"[INFO] Found {len(configs)} config.yml file(s)")

    n_changed = 0
    for cfg in configs:
        if rewrite_config(cfg, project_root, args.dry_run):
            n_changed += 1

    print()
    if args.dry_run:
        print(f"[INFO] Dry run: would rewrite {n_changed}/{len(configs)} file(s)")
    else:
        print(f"[INFO] Rewrote {n_changed}/{len(configs)} file(s)")
        print(f"[INFO] Backups saved as <config>.yml.bak")


if __name__ == "__main__":
    main()