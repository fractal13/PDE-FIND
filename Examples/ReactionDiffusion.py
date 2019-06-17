#!/usr/bin/env python3

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PDE_FIND import *
import scipy.io as sio
import itertools

data = sio.loadmat('./canonicalPDEs/reaction_diffusion.mat')

