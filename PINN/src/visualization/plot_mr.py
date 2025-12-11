# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:12:47 2025

@author: zhang
"""


import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from config_hub2 import _importpath
# sys.path.append(_importpath)

def plot_mass_radius(model):
    r_test = np.linspace(0.01, 15, 100).reshape(-1, 1).astype(np.float32)
    predictions = model.predict(r_test)
    p_pred, m_pred = predictions[:, 0], predictions[:, 1]

    plt.figure(figsize=(8, 6))
    plt.plot(r_test, m_pred)
    plt.xlabel("Radius (km)")
    plt.ylabel("Mass (M☉)")
    plt.title("Neutron Star Mass-Radius Relation")
    plt.grid()
    plt.show()