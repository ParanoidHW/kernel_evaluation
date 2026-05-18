from __future__ import annotations

import sys

from op_eval.cli import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
