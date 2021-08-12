#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:26:17 2021

@author: soominkwon
"""

import numpy as np
import matplotlib.pyplot as plt
from reshaped_wirtinger_flow import rwf_fit
from generate_rwf import generateOneMagnitude

# initializing
image_name = 'image_tensor_small.npz'
m_dim = 3000 # 10x the dimensions of the image

# generating sample data
x_reshaped, y, A_mat = generateOneMagnitude(image_name=image_name, m_dim=m_dim)

# solving for x using RWF
x_solved = rwf_fit(y=y, A=A_mat)

# reshaping solution for plotting
x_solved_reshaped = np.reshape(x_solved, (x_reshaped.shape))

# plotting solutions
plt.imshow(np.abs(x_reshaped), cmap='gray')
plt.title('True Image')
plt.show()

plt.imshow(np.abs(x_solved_reshaped), cmap='gray')
plt.title('Solved Image via RWF')
plt.show()


