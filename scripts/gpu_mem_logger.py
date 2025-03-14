"""
This is a simple script to log the GPU memory usage of the current GPU.
It's useful to monitor how much memory the inference models are using when running on an interactive node.
It will log the memory usage to a file called `gpu.log` in the same directory.
Activate it with `python gpu_mem_logger.py`, send it to the background with `Ctrl+Z` and `bg`, and stop it with `kill %1`.
"""

import subprocess
import time
from datetime import datetime

cmd = 'nvidia-smi --query-gpu=memory.used --format=csv'
prev_mem = None
while True:
    with open("gpu.log", "a+") as f:
        out = subprocess.run(cmd.split(), capture_output=True)
        txt = out.stdout.decode('utf-8')
        mem = txt.split("\n")[1]
        if mem != prev_mem:
            f.write(f"MEM @ {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}: {mem}\n")
            prev_mem = mem
        time.sleep(0.2)
