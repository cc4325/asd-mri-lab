"""ASD/HC T1 MRI 教学项目。"""

__version__ = "0.1.0"

import os
from pathlib import Path

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "outputs" / ".mplcache"))

