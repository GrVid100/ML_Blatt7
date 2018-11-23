import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df[['class']].values

# Standardize the Scale
from sklearn.preprocessing import StandardScaler
X_standardized = StandardScaler().fit_transform(X)

# Remove Mean
mean = np.mean(X_standardized)
X_standardized = X_standardized - mean

# Calculate Sample Variance
covariance_matrix = np.cov(X_standardized.T)

# Calculate Eigenvalue Decomposition
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Order Eigenvalues
eigen_values_with_eigen_vectors = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
eigen_values_with_eigen_vectors.sort()
eigen_values_with_eigen_vectors.reverse()

# Choose two biggest Eigenvalues
V = np.vstack((eigen_values_with_eigen_vectors[0][1],
               eigen_values_with_eigen_vectors[1][1]))

# Project Data onto V
Z = V.dot(X_standardized.T)

x_values = Z[0, :]
y_values = Z[1, :]

# Plot Data
colors = []
for index in range(X.shape[0]):
    if y[index] == 'Iris-setosa':
        colors.append('green')
    elif y[index] == 'Iris-versicolor':
        colors.append('red')
    else:
        colors.append('blue')

plt.scatter(x_values, y_values, c=colors)
plt.savefig('scatter_plot')
plt.show()
