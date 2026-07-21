import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import io

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib


matplotlib.use("Agg")  # non-interactive backend; avoids tkinter main-loop errors
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import warnings
