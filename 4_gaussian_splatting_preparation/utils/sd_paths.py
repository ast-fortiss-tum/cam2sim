"""
Resolve where the cam2sim Stable Diffusion models live.

The chosen path is persisted in `data/.sd_root` (project root).

Resolution order (per call):
  1. If `override` is given (e.g. --output-root from 4A), use it.
     The value is written to `data/.sd_root` so later runs find it
     without the flag.
  2. Else, if `data/.sd_root` exists, read and use the path from it.
  3. Else, fall back to <PROJECT_ROOT>/data/stable_diff_models and
     persist it as the chosen path. This makes the project work
     out of the box right after `git clone`, without forcing the
     user to configure anything.

The returned path is always absolute. The directory is created if
it does not exist.
"""

import os

SD_ROOT_FILE = "data/.sd_root"
DEFAULT_SD_ROOT_REL = "data/stable_diff_models"


def resolve_sd_root(project_root, override=None):
    """
    Returns (path, action_label):
      - path:         absolute path to the SD root
      - action_label: short string for logging ("--output-root",
                      "data/.sd_root", "default")
    """
    path_file = os.path.join(project_root, SD_ROOT_FILE)

    if override is not None:
        chosen = os.path.abspath(override)
        _persist(path_file, chosen)
        action = "--output-root"
    elif os.path.exists(path_file):
        with open(path_file, "r") as f:
            chosen = f.read().strip()
        if not chosen:
            raise ValueError(f"SD root file is empty: {path_file}")
        action = SD_ROOT_FILE
    else:
        chosen = os.path.abspath(os.path.join(project_root, DEFAULT_SD_ROOT_REL))
        _persist(path_file, chosen)
        action = f"default ({DEFAULT_SD_ROOT_REL})"

    os.makedirs(chosen, exist_ok=True)
    return chosen, action


def _persist(path_file, path):
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    with open(path_file, "w") as f:
        f.write(path + "\n")