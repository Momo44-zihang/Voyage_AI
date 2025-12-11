# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:13:56 2025

@author: zhang
"""


import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config_hub_ai import _importpath
sys.path.append(_importpath)
from PINN.src.models import tov_pinn
from PINN.src.training import train 
from PINN.src.visualization import plot_mr

model = tov_pinn.TOV_PINN_with_IC()
train.train_pinn(model, epochs=2000, learning_rate=1e-3)
plot_mr.plot_mass_radius(model)
