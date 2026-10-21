"""One-command offline CPU demonstration and non-optimizing tests."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"))
    for arguments in (("tools/demo_supported_comparison.py",), ("tools/demo_language_model.py",),
                      ("-m", "pytest", "-q", "-p", "no:cacheprovider")):
        subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env, check=True)

if __name__ == "__main__":
    main()
