"""Correct MuJoCo 2.3.7's Apple OpenGL path inside the active virtualenv."""

import ctypes
import sys
import sysconfig
from pathlib import Path

if sys.platform == "darwin":
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this script with the project's virtualenv Python.")
    target = Path(sysconfig.get_path("purelib")) / "mujoco/cgl/cgl.py"
    old = "/System/Library/OpenGL.framework/OpenGL"
    new = "/System/Library/Frameworks/OpenGL.framework/OpenGL"
    ctypes.CDLL(new)
    source = target.read_text()
    if old in source:
        target.write_text(source.replace(old, new))
        print("Corrected MuJoCo's macOS OpenGL framework path.")
    else:
        print("MuJoCo's macOS OpenGL path needs no correction.")
