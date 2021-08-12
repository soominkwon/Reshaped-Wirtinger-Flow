# Reshaped-Wirtinger-Flow
Reshaped Wirtinger Flow (RWF) implementation for solving complex valued signals. This implementation is based on the paper "Reshaped Wirtinger Flow and Incremental Algorithms for Solving Quadratic Systems of Equations."

For more information: https://arxiv.org/abs/1605.07719


## Programs
The following is a list of which algorithms correspond to which Python script:

* generate_rwf.py - Generates sample measurements for testing
* image_tensor_small.npz - Sample image
* reshaped_wirtinger_flow.py - Implementation of RWF
* rwf_run.py - Example on using RWF implementation

## Tutorial
This tutorial can be found in rwf_run.py:

```
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
```

## Solution Example

<p align="center">
  <a href="url"><img src="https://github.com/soominkwon/Reshaped-Wirtinger-Flow/blob/main/rwf_example.png" align="left" height="300" width="300" ></a>
</p>

