import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

## Génération d'un jeu de données
np.random.seed(0) # Permet d'avoir à chaque fois les mêmes données générées

# Définition de nos centroïdes
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers) # Nb centroïdes
# Génération des données avec make_blobs
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
# labels_true correspond à l'assignation des points aux labels
# c-à-d à quel cluster les points doivent appartenir

## Lancement d'un méthode de clustering avec les kmeans
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

## Affichage du résultat avec matplotlib
fig = plt.figure(1, figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#F92672', '#66D9EF', '#A6E22E']

ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k] # centroïdes
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, 
        marker='.') # affichage des points
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        markeredgecolor='k', markersize=6) # affichage des centroïdes

ax.set_title("Classification avec l'algorithme des KMeans de sklearn")
ax.set_xticks(())
ax.set_yticks(())
# Affichage de l'inertie
plt.text(-3.5, 1.8, "Inertie : %.3f"%(k_means.inertia_))
plt.show() # Output du graphique
