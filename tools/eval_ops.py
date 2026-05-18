#!/usr/bin/env python3
"""Evaluate operator profiling rows with kernel-aware analytic models."""

from __future__ import annotations

import sys

from op_eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
