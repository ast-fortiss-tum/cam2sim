#!/usr/bin/env python3
"""
fix_config_paths.py

After unzipping data.zip, rewrite absolute paths inside every
splatfacto config.yml to match the current project root.

Nerfstudio embeds full paths in config.yml (output_dir, data, load_dir, ...)
which break when the bundle is unzipped on a different machine.

Usage (from project root):
    python 4_gaussian_splatting/fix_config_paths.py
"""

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

GS_OUTPUTS_GLOB = "data/data_for_gaussian_splatting/*/outputs/splatfacto_split_*/splatfacto/*/config.yml"


def find_path_keys(yaml_text):
    """
    Return list of (key, abs_path) for every line that looks like:
        key: /some/absolute/path/...
    """
    pattern = re.compile(r"^(\s*\w+):\s*(/[^\s]+)\s*$", re.MULTILINE)
    return [(m.group(1), m.group(2)) for m in pattern.finditer(yaml_text)]


def rewrite_config(config_path: Path, new_root: Path, dry_run: bool):
    text = config_path.read_text()
    original = text

    # Match any absolute path that contains "data/data_for_gaussian_splatting/..."
    # and rewrite the prefix before it to the new project root.
    pattern = re.compile(r"(/[^\s]*?)(data/data_for_gaussian_splatting/[^\s]+)")

    def replace(m):
        return f"{new_root}/{m.group(2)}"

    text = pattern.sub(replace, text)

    if text == original:
        print(f"[OK ] {config_path}  (no changes)")
        return False

    if dry_run:
        print(f"[DRY] {config_path}  (would be rewritten)")
    else:
        config_path.write_text(text)
        print(f"[FIX] {config_path}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would change without writing.")
    args = parser.parse_args()

    print(f"[INFO] Project root: {PROJECT_ROOT}")

    configs = sorted(PROJECT_ROOT.glob(GS_OUTPUTS_GLOB))
    if not configs:
        print(f"[ERROR] No config.yml found under "
              f"{GS_OUTPUTS_GLOB}")
        sys.exit(1)

    print(f"[INFO] Found {len(configs)} config.yml file(s)")
    n_changed = 0
    for cfg in configs:
        if rewrite_config(cfg, PROJECT_ROOT, args.dry_run):
            n_changed += 1

    print()
    if args.dry_run:
        print(f"[INFO] Dry run: would rewrite {n_changed}/{len(configs)} file(s)")
    else:
        print(f"[INFO] Rewrote {n_changed}/{len(configs)} file(s)")


if __name__ == "__main__":
    main()