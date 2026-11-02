"""VLAC convenience entry point; the common launcher owns process lifetimes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_libero import main

if __name__ == "__main__":
    main(["--method", "vlac", *sys.argv[1:]])
