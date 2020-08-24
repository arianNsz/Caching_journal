# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:42:18 2020

@author: arian92
"""

import numpy as np

req = np.genfromtxt('./data/challenging_popularity.csv', delimiter=',')
req = np.delete(req, 0, 0)

ind = list(range(int(len(req)/2)))

ind = np.array(ind)*2

edge1 = req[ind]

edge2 = req[ind+1]

np.savetxt('./data/edge1.csv', edge1, delimiter=',')
np.savetxt('./data/edge2.csv', edge2, delimiter=',')