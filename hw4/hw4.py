# Scientific Concepts and Thinking
# HW 4: Network Thinking and Linear Algebra and All That
# Submitted by Vanessa Tan 20225640

import numpy as np
import networkx as nx


## Lifting yourself up by your own bootstrap
print('Lifting yourself up by your own bootstrap')
AA = np.matrix([[0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 1, 1],
                [0, 1, 1, 0, 1, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0]])

x = np.matrix([0.257,0.206,0.576,0.53,0.462,0.257]).T
xt = np.matmul(AA, x)

x_norm = x / np.linalg.norm(x)

eigen = np.matrix([0.245,0.212,0.599,0.518,0.457,0.245])
corr = np.corrcoef(x.T, eigen)
print('Correlation:\n' + str(corr) + '\n')


## Eigenvalue vs Katz Centralities
print('Eigenvalue vs Katz Centralities')
A = np.matrix([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0]])
w, v = np.linalg.eig(A)
print('Eigenvalues:\n' + str(w))
print('Eigenvectors:\n' + str(v))

print('\nEigenvalue Centralities')
eig = nx.from_numpy_matrix(A, create_using=nx.Graph)
eig_centrality = nx.eigenvector_centrality_numpy(eig.copy())
print(['node(%s) %f'%(node,eig_centrality[node]) for node in eig_centrality])

print('\nKatz Centralities')
katz = nx.from_numpy_matrix(A, create_using=nx.Graph)
for i in [0, 0.5, 0.85, 1, 2]:
    katz_centrality = nx.katz_centrality_numpy(katz, alpha=i, beta=1, normalized=True)
    print('alpha: ' + str(i))
    print(['node(%s) %f' % (node, katz_centrality[node]) for node in katz_centrality])

## The Three Centralities
print('\nThe Three Centralities')
print('Page Rank')
pr = nx.from_numpy_matrix(AA, create_using=nx.Graph)
pagerank = nx.pagerank(pr, alpha=0.85)
print(['node(%s) %f' % (node, pagerank[node]) for node in pagerank])

k = np.matrix([1,1,4,3,2,1])
e = np.matrix([0.245,0.212,0.599,0.518,0.457,0.245])
p = np.matrix([0.093,0.093,0.32,0.241,0.161,0.093])
print('k & e (Corr):\n' + str(np.corrcoef(k, e)))
print('k & p (Corr):\n' + str(np.corrcoef(k, p)))
print('e & p (Corr):\n' + str(np.corrcoef(e, p)))