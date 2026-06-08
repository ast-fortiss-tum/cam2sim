#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nerfstudio.py

Shared helpers for the Nerfstudio pipeline
(5C, 5D, and anything else that needs to discover trained nerfstudio models).

Currently contains:
  - SUPPORTED_METHODS: canonical list of training methods we support.
  - detect_available_methods(): cheap scan of the outputs directory to
    figure out which methods have at least one trained split.
"""

import os
import re


# Methods supported by the 4B training step. Nerfstudio writes:
#   data/data_for_gaussian_splatting/<BAG>/outputs/<METHOD>_split_<N>/<METHOD>/<TS>/config.yml
# Order matters: it is the canonical display/return order used by
# detect_available_methods() and other helpers in this module.
SUPPORTED_METHODS = (
    "splatfacto",
    "splatfacto-big",
    "nerfacto",
    "nerfacto-big",
)


def detect_available_methods(gs_outputs_dir, supported_methods=SUPPORTED_METHODS):
    """
    Scan gs_outputs_dir for subfolders matching <method>_split_<N>, and
    return the list of methods (full names, longest-match) that have at
    least one such folder.

    Does NOT validate the inner config.yml / utm_transform.json files: it
    is a cheap pre-check, meant to be called BEFORE deciding whether to
    auto-select a single method or ask the user to disambiguate with
    --method. Full validation happens later in auto_detect_splits().

    Match is longest-first, so a folder named "splatfacto-big_split_1"
    is classified as "splatfacto-big" and never as "splatfacto".

    Returns the methods in the canonical order of `supported_methods`,
    not in filesystem-listing order.
    """
    if not os.path.isdir(gs_outputs_dir):
        return []

    methods_by_length = sorted(supported_methods, key=len, reverse=True)
    patterns = {
        m: re.compile(rf"^{re.escape(m)}_split_(\d+)$")
        for m in supported_methods
    }

    found = set()
    for entry in os.listdir(gs_outputs_dir):
        if not os.path.isdir(os.path.join(gs_outputs_dir, entry)):
            continue
        for m in methods_by_length:
            if patterns[m].match(entry):
                found.add(m)
                break  # longest match wins, do not fall through

    return [m for m in supported_methods if m in found]