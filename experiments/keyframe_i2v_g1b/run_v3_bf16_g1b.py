#!/usr/bin/env python
"""BF16 runtime shim for the official V3 arbitrary-frame script.

Environment-only adjustment: PyTorch default dtype -> bfloat16 before the
official script runs, so diffusers loads the 14B checkpoint in BF16
(~33GB GPU instead of ~65GB FP32). No AniSora source is modified.
"""
import os
import runpy
import sys

import torch

torch.set_default_dtype(torch.bfloat16)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
sys.argv = ["generate-pi-i2v-any.py"] + sys.argv[1:]
runpy.run_path(
    os.path.join(os.getcwd(), "generate-pi-i2v-any.py"),
    run_name="__main__",
)
