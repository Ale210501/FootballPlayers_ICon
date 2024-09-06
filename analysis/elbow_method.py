import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Carica il dataset preprocessato
cluster_dataset = pd.read_csv('../dataset/players_preprocessato.csv')

# INIZIO CLUSTERING
# METODO DEL GOMITO
wcss = []

# Calcola il WCSS per un numero di cluster da 1 a 9
for i in range(1, 10):
    kmeansOut = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeansOut.fit(cluster_dataset)
    wcss.append(kmeansOut.inertia_)

# Tracciamo il risultato su un grafico a linee per osservare 'Il gomito'
plt.plot(range(1, 10), wcss, 'bx-')
plt.title('Metodo del gomito')
plt.xlabel('Numero di cluster')
plt.xticks(range(1, 10))
plt.ylabel('WCSS')  # Somma dei quadrati entro i cluster
plt.grid(True)
plt.show()

# Numero di cluster scelto dopo l'osservazione del metodo del gomito
k = 3

# Applica k-means al dataset / Crea il classificatore k-means
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Prevedi i cluster per ogni punto del dataset
y_kmeans = kmeans.fit_predict(cluster_dataset)

# Aggiunge la colonna 'cluster' al dataset, con i cluster numerati da 1
cluster_dataset["cluster"] = kmeans.labels_ + 1
