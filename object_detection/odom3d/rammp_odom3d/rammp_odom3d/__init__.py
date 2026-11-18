"""odom3d: world-anchored object tracking (see core.py)."""

import os

# The tracker's matrix ops are tiny. A multi-threaded OpenBLAS pool busy-waits between them: on the bag replay the
# node took 2.8 cores instead of 0.17. Set before numpy loads (this module runs first); an explicit value wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
