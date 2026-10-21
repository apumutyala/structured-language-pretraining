"""Portable fresh build of the exported Lean 4.8.0 specifications; CPU only."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", default="lean", help="Lean 4.8.0 executable on PATH or explicit path")
    args=parser.parse_args(); executable=shutil.which(args.lean)
    if executable is None:
        raise SystemExit("Lean executable unavailable; install Lean 4.8.0 or pass --lean")
    version=subprocess.check_output([executable,"--version"],text=True).strip()
    if "version 4.8.0" not in version:
        raise SystemExit("Exact Lean 4.8.0 required: "+version)
    output=ROOT / ".build/lean";output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,LEAN_PATH=str(output))
    for name in ("SoundMemory","SoundMemoryReview","WorkflowKernel","WorkflowKernelCli"):
        subprocess.run([executable,"-o",str(output/(name+".olean")),
            str(ROOT/"proofs/certified_memory"/(name+".lean"))],env=env,check=True)
    print("Fresh CPU build complete: "+version)
    print("For the differential tool, set WORKFLOW_LEAN to this executable path.")

if __name__=="__main__":main()
