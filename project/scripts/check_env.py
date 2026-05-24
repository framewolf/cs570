import os
import torch
import numpy as np
import scipy
import h5py
import cv2
import robomimic
import robosuite

print("ROBOMIMIC_DIR:", os.environ.get("ROBOMIMIC_DIR"))
print("ROBOSUITE_DIR:", os.environ.get("ROBOSUITE_DIR"))
print("MUJOCO_GL:", os.environ.get("MUJOCO_GL"))

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)

print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("h5py import OK")
print("cv2:", cv2.__version__)
print("robomimic import OK")
print("robosuite import OK")
