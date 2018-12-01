import random
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph

random.seed(1234)

# Aufgabenteil a)
X, colors = datasets.samples_generator.make_swiss_roll(800)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap=plt.cm.Spectral)
ax.set_title('Original Data')
plt.savefig("original_data")
plt.show()

# Aufgabenteil b)

# Remove Mean
X = X - np.mean(X)

# Calculate Sample Variance
covariance_matrix = np.cov(X.T)

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
Z = V.dot(X.T)

x_values = Z[0, :]
y_values = Z[1, :]

# Plot Data
fig, ax = plt.subplots()
ax.set_title("Aufgabe 2, Aufgabenteil b) - Data mit PCA")
ax.scatter(x_values, y_values, c=colors, cmap=plt.cm.Spectral)
plt.savefig('plot_aufgabe2_b')
plt.show()


# Die "mds" Funktion wurde aus der Vorlesung übernommen
def mds(A, d = 2):
    D = A**2

    G = D - D.mean(axis=0)
    G = D - D.mean(axis=1)
    G = -0.5*G

    G = (G+G.T)/2

    (Lambda, V) = np.linalg.eig(G)

    pairs = sorted(zip(Lambda, V.T),
                   key=lambda x: x[0].real, reverse=True)

    Lambda = [d[0] for d in pairs[0:d]]
    V = np.array([d[1] for d in pairs[0:d]])

    R = np.diag(np.sqrt(Lambda))@V
    return R.T


# Aufgabenteil d) und f)

# Mit k-nearest Neighbor
D = kneighbors_graph(X, 8)
D = sparse.csgraph.shortest_path(D, 'FW')

X_recunstructed = mds(D, 3)

fig, ax = plt.subplots()
ax.set_title("ISOMAP mit k-nearest Neigbor")
ax.scatter(X_recunstructed[:, 0],
           X_recunstructed[:, 2], c=colors, cmap=plt.cm.Spectral)
plt.savefig("ISOMAP_mit_knn")
plt.show()

# Mit Epsilon
D = radius_neighbors_graph(X, 5)
D = sparse.csgraph.shortest_path(D, 'FW')

X_recunstructed = mds(D, 3)

fig, ax = plt.subplots()
ax.set_title("ISOMAP mit Epsilon")
ax.scatter(X_recunstructed[:, 0],
           X_recunstructed[:, 2], c=colors, cmap=plt.cm.Spectral)
plt.savefig("ISOMAP_mit_epsilon")
plt.show()

