#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:54:47 2021

@author: soominkwon
"""

import numpy as np


def generateOneMagnitude(image_name, m_dim):
    """
        Function to demonstrate and prepare data for RWF.
        
        Returns:
            x: True image
            y: Observations (m x 1)
            A_mat: Sampling matrix (under real Gaussian design)
    
    """
    with np.load(image_name) as data:
        tensor = data['arr_0']
        
    first_image = tensor[:, :, 0]
    norm = np.linalg.norm(first_image, 'fro')
    first_image = first_image / norm
    vec_image = np.reshape(first_image, (-1, 1))
    
    n_dim = vec_image.shape[0]
    
    A_mat = np.random.randn(n_dim, m_dim)
    y_measurements = A_mat.T @ vec_image
    
    y = np.abs(y_measurements)
    
    return first_image, y, A_mat

    