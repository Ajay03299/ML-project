"""Kept for old commands. Same as `python -m experiments.run_benchmark` (see that file)."""
import sys

from experiments.run_benchmark import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--only" in argv:                       # accept the old comma-separated form too
        i = argv.index("--only")
        if i + 1 < len(argv) and "," in argv[i + 1]:
            argv = argv[:i + 1] + argv[i + 1].split(",") + argv[i + 2:]
    main(argv)
