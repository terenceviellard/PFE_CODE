import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def compute_bic(gmm, X):
    return gmm.bic(X)

def find_best_k(X, max_k=10):
    bic_values = []

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            n_init=20,
            max_iter=2000,
            random_state=42
        )
        gmm.fit(X)

        if not gmm.converged_:
            print(f"⚠️ Model did not converge for k={k}.")

        bic = compute_bic(gmm, X)
        bic_values.append(bic)

        print(f"k={k} -> BIC: {bic}")
        print(f"Weights for k={k}: {np.round(gmm.weights_, 3)}")
        labels = gmm.predict(X)
        unique_clusters = np.unique(labels)
        print(f"Clusters détectés pour k={k}: {unique_clusters}")

    best_k_bic = np.argmin(bic_values) + 1

    return best_k_bic, bic_values

# Chargement des données
data = pd.read_csv('data/data_pfe_n1000_L10.csv')
X = data[["V1", "V2"]].values

# Trouver le meilleur k
best_k_bic, bic_values = find_best_k(X, max_k=10)
print(f"Best k (BIC): {best_k_bic}")

# Utiliser le meilleur modèle pour prédire les clusters
best_gmm = GaussianMixture(
    n_components=best_k_bic,
    covariance_type='full',
    n_init=20,
    max_iter=2000,
    random_state=42
)
best_gmm.fit(X)
labels = best_gmm.predict(X)

# Ajouter les labels de clusters aux données
data['Cluster'] = labels

# Créer les répertoires de sortie s'ils n'existent pas
output_dir = 'output/results/bgmm'
os.makedirs(output_dir, exist_ok=True)

# Sauvegarder les résultats dans un fichier CSV
data.to_csv(os.path.join(output_dir, 'cluster_assignments_bgmm.csv'), index=False)

# Sauvegarde des résultats BIC
results_df = pd.DataFrame({
    'k': range(1, len(bic_values) + 1),
    'BIC': bic_values
})
results_df.to_csv(os.path.join(output_dir, 'BIC_bgmm.csv'), index=False)

# Visualisation des critères
plt.figure(figsize=(8, 5))
plt.plot(results_df['k'], results_df['BIC'], marker='o', label='BIC', color='green')
plt.xlabel("Nombre de clusters k")
plt.ylabel("Valeur des critères")
plt.legend()
plt.title("Sélection du nombre optimal de clusters avec BIC")
plt.grid()
plt.savefig(os.path.join(output_dir, 'bic_plot.png'))  # Sauvegarde du graphique
plt.show()

# Visualisation et sauvegarde du meilleur modèle
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.title(f'Clusters détectés pour le meilleur k={best_k_bic}')
plt.savefig(os.path.join(output_dir, 'best_model_clusters.png'))  # Sauvegarde du graphique
plt.show()
