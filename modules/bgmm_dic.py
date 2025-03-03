import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

def compute_dic(bgmm, X):
    log_likelihood = bgmm.score(X) * X.shape[0]
    deviance = -2 * log_likelihood

    responsibilities = bgmm.predict_proba(X)
    log_prob_norm = bgmm._estimate_weighted_log_prob(X)
    expected_log_likelihood = np.sum(responsibilities * log_prob_norm)

    p_d = 2 * (log_likelihood - expected_log_likelihood)
    dic = deviance + 2 * p_d
    return dic

def compute_waic(bgmm, X):
    log_prob = bgmm._estimate_weighted_log_prob(X)
    epsilon = 1e-10  # Évite log(0)
    lppd = np.sum(np.log(np.mean(np.exp(log_prob) + epsilon, axis=0)))
    p_waic = np.sum(np.var(log_prob, axis=0))
    waic = -2 * (lppd - p_waic)
    return waic

def find_best_k(X, max_k=10, penalty_factor=0.1):
    dic_values = []
    waic_values = []

    for k in range(1, max_k + 1):
        bgmm = BayesianGaussianMixture(
            n_components=k,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1e-4,  # Réduit pour favoriser plus de clusters
            covariance_type='full',
            n_init=20,
            max_iter=2000,
            random_state=42
        )
        bgmm.fit(X)

        if not bgmm.converged_:
            print(f"⚠️ Model did not converge for k={k}.")

        dic = compute_dic(bgmm, X)
        waic = compute_waic(bgmm, X)

        # Ajouter une pénalité pour les modèles avec peu de clusters
        dic += penalty_factor * (max_k - k)
        waic += penalty_factor * (max_k - k)

        dic_values.append(dic)
        waic_values.append(waic)

        print(f"k={k} -> DIC: {dic}, WAIC: {waic}")
        print(f"Weights for k={k}: {np.round(bgmm.weights_, 3)}")
        labels = bgmm.predict(X)
        unique_clusters = np.unique(labels)
        print(f"Clusters détectés pour k={k}: {unique_clusters}")

        # Visualisation des clusters
        #plt.figure(figsize=(6, 4))
        #plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
        #plt.title(f'Clusters détectés pour k={k}')
        #plt.show()

    best_k_dic = np.argmin(dic_values) + 1
    best_k_waic = np.argmin(waic_values) + 1

    return best_k_dic, best_k_waic, dic_values, waic_values

# Chargement des données
data = pd.read_csv('data/data_pfe_n1000_L10.csv')
X = data[["V1", "V2"]].values

# Trouver le meilleur k
best_k_dic, best_k_waic, dic_values, waic_values = find_best_k(X, max_k=10)
print(f"Best k (DIC): {best_k_dic}, Best k (WAIC): {best_k_waic}")

# Sauvegarde des résultats
results_df = pd.DataFrame({
    'k': range(1, len(dic_values) + 1),
    'DIC': dic_values,
    'WAIC': waic_values
})

# Visualisation des critères
plt.figure(figsize=(8, 5))
plt.plot(results_df['k'], results_df['DIC'], marker='o', label='DIC', color='red')
plt.plot(results_df['k'], results_df['WAIC'], marker='s', label='WAIC', color='blue')
plt.xlabel("Nombre de clusters k")
plt.ylabel("Valeur des critères")
plt.legend()
plt.title("Sélection du nombre optimal de clusters avec DIC et WAIC")
plt.grid()
plt.show()
