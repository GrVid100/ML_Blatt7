import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

X = [210.0, 209.0, -116.0, -249.0, 174.0, 190.0,
     277.0, 89.0, -105.0, -189.0, 79.0, 82.0,
     -66.0, 122.0, -13.0, -59.0, 95.0, 107.0,
     486.0, 296.0, -219.0, -439.0, 253.0, 273.0]

X = np.array(X)
X = X.reshape(4, 6)

# Die Dimensionen sind verschiedene Messpunkte. Deswegen möchten wir
# Messpunkte als Spalten haben und transponieren die Matrix.
X = X.T

# Standardize the Scale
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

# Remove Mean
mean = np.mean(X)
X_standardized = X - mean

# Calculate Sample Variance
covariance_matrix = np.cov(X_standardized.T)

# Calculate Eigenvalue Decomposition
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Order Eigenvalues
eigen_values_with_eigen_vectors = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
eigen_values_with_eigen_vectors = sorted(eigen_values_with_eigen_vectors, key=itemgetter(0))
eigen_values_with_eigen_vectors.reverse()

eigenvalues_list = [eigen_value_with_eigen_vector[0] for eigen_value_with_eigen_vector in eigen_values_with_eigen_vectors]
print(eigenvalues_list)

# Wir sehen, dass die ersten Eigenvalues viel größer sind als die restlichen.
# Da dieser Wert über 90% des Traces ausmacht, nehmen wir nur zwei Dimensionen.

V = np.vstack((eigen_values_with_eigen_vectors[0][1],
               eigen_values_with_eigen_vectors[1][1]))

# Project Data onto V
Z = V.dot(X_standardized.T)

x_values = [measurement for measurement in range(6)]
y_values1 = Z[0, :]
plt.scatter(x_values, y_values1)

y_values2 = Z[1, :]
plt.scatter(x_values, y_values2)

plt.savefig('plot_aufgabe1')
plt.show()

