# Functions to read and show images.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh 

d0 = pd.read_csv('mnist_train.csv')

print(d0.head(5)) # print first five rows of d0.

# Save the labels into a variable named 'l'.
l = d0['label']

# Drop the label feature and store the pixel data in variable 'd'.
d = d0.drop("label", axis=1)
print(d.shape)
print(l.shape)

def display(data):
    # Display or plot a number.
    plt.figure(figsize=(7, 7))
    idx = 1

    grid_data = data.iloc[idx].to_numpy().reshape(28, 28)  # Reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation="none", cmap="gray")
    plt.show()

    print(l[idx])

# Uncomment the following line to call the display() function.
# display(d)

def label(data):
    # Pick the first 15K data-points to work on for time efficiency.
    # Exercise: Perform the same analysis on all 42K data-points.

    labels = l.head(15000)
    data = data.head(15000)

    print("The shape of sample data = ", data.shape)

    # Data-preprocessing: Standardizing the data
    standardized_data = StandardScaler().fit_transform(data)
    print(standardized_data.shape)

    # Find the co-variance matrix which is: A^T * A
    sample_data = standardized_data

    # Matrix multiplication using numpy
    covar_matrix = np.matmul(sample_data.T , sample_data)

    print("The shape of variance matrix = ", covar_matrix.shape)

    # Finding the top two eigen-values and corresponding eigen-vectors for projecting onto a 2-Dim space.

    # The parameter 'eigvals' is defined (low value to high value). 
    # The eigh function will return the eigen values in ascending order.
    # This code generates only the top 2 (782 and 783) eigenvalues.
    values, vectors = eigh(covar_matrix, eigvals=(782, 783))

    print("Shape of eigen vectors = ", vectors.shape)
    # Converting the eigen vectors into (2,d) shape for ease of further computations.
    vectors = vectors.T

    print("Updated shape of eigen vectors = ", vectors.shape)
    # Here, vectors[1] represents the eigen vector corresponding to the 1st principal eigen vector,
    # and vectors[0] represents the eigen vector corresponding to the 2nd principal eigen vector.

    return vectors, values

eigen_vectors, eigen_values = label(d)

# You can perform further computations or analysis using the obtained eigen vectors and eigen values.
