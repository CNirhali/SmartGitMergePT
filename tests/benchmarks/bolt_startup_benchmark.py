import time
import os
import sys
import subprocess

def benchmark():
    # Measure import time of src/agentic_tracker.py
    # We use a separate process to get a clean state
    cmd = [sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); import time; s=time.time(); import agentic_tracker; print(time.time()-s)"]

    # Mocking dependencies if they are missing to allow the script to run
    # but the goal is to measure the REAL import time if they ARE present.
    # Since they are missing in this environment, it's fast anyway.

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        print(f"Import time for agentic_tracker: {output}s")
    except subprocess.CalledProcessError as e:
        print(f"Error measuring import time: {e.output.decode()}")

if __name__ == "__main__":
    benchmark()
