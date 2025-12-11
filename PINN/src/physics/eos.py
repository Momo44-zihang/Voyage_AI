# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:06:15 2025

@author: zhang
"""

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from config_hub2 import _importpath
# sys.path.append(_importpath)

# 设置常量
anr = 2.4216
ar = 2.8663

def eos(p):
    epsilon = anr * p ** (3 / 5) + ar * p
    return epsilon
