from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from playerposprediction import *


def reformat_player(row, player):
    # Riformatta il nome del giocatore utilizzando il dizionario player_dic
    for key in player_dic:
        if player_dic[key] == row[player]:
            row[player] = key
            return row[player]


def similarity_with_cosine(row_a, row_b):
    # Calcola la similarità coseno tra due righe
    element = cosine_similarity([row_a], [row_b])
    row_a['similarity'] = element[0][0]
    return row_a['similarity']


def set_recommender(ast, pos, matchplayed, crdy, crdr, gls, player, born):
    # Carica i dati dei giocatori e il dizionario di conversione
    cluster_dataset = pd.read_csv('../dataset/players_preprocessato.csv')
    conversion = pd.read_csv('../dataset/dizionario.csv')

    # Crea il dizionario di conversione
    conversion.apply(lambda row: set_dictionary(row), axis=1)

    # Prepara i dati dell'utente per la raccomandazione
    row_user = []

    # Rimuove le colonne non utilizzate se il valore è vuoto
    if pos == '':
        cluster_dataset = cluster_dataset.drop(columns=['Pos_format'])
    else:
        row_user.append(int(search_format_pos(pos)))

    if ast == '':
        cluster_dataset = cluster_dataset.drop(columns=['Ast'])
    else:
        row_user.append(int(ast))

    if gls == '':
        cluster_dataset = cluster_dataset.drop(columns=['Gls'])
    else:
        row_user.append(int(gls))

    if player == '':
        cluster_dataset = cluster_dataset.drop(columns=['Player_format'])
    else:
        row_user.append(int(search_format_player(player)))

    if matchplayed == '':
        cluster_dataset = cluster_dataset.drop(columns=['90s'])
    else:
        row_user.append(int(matchplayed))

    if crdy == '':
        cluster_dataset = cluster_dataset.drop(columns=['CrdY'])
    else:
        row_user.append(int(crdy))

    if crdr == '':
        cluster_dataset = cluster_dataset.drop(columns=['CrdR'])
    else:
        row_user.append(int(crdr))

    if born == '':
        cluster_dataset = cluster_dataset.drop(columns=['Born'])
    else:
        row_user.append(float(born))

    # Normalizza i dati dell'utente
    row_user_normalized = preprocessing.normalize([row_user])

    # Esegue il clustering dei dati dei giocatori
    kmeans = KMeans(n_clusters=3).fit(preprocessing.normalize(cluster_dataset))

    # Assegna i cluster ai dati dei giocatori
    cluster_dataset["cluster"] = kmeans.labels_

    # Predizione del cluster per i dati dell'utente
    prediction = kmeans.predict(row_user_normalized)

    # Seleziona il cluster dell'utente
    user_cluster = prediction[0]
    split_cluster = cluster_dataset[cluster_dataset['cluster'].apply(lambda x: x == user_cluster)]

    # Rimuove la colonna 'cluster'
    split_cluster = split_cluster.drop(columns=['cluster'])

    # Calcola la similarità tra i dati dell'utente e i dati dei giocatori
    split_cluster['similarity'] = split_cluster.apply(lambda row: similarity_with_cosine(row, row_user), axis=1)

    # Ordina i giocatori in base alla similarità
    split_cluster.sort_values(['similarity'], ascending=False, inplace=True)

    # Imposta il dizionario dei giocatori
    set_player()

    # Seleziona i primi 10 giocatori più simili
    ten_sim = split_cluster.head(10)

    # Riformatta i nomi dei giocatori
    ten_sim = ten_sim.loc[:, ['Player_format', 'similarity']]
    ten_sim['Player_format'] = ten_sim.apply(lambda row: reformat_player(row, 'Player_format'), axis=1)

    # Ordina i giocatori in base alla similarità
    ten_sim.sort_values(['similarity'], ascending=False, inplace=True)

    # Stampa i giocatori suggeriti
    print("I giocatori suggeriti in base alle tue preferenze sono:\n")

    i = 1
    for element in ten_sim['Player_format']:
        print(f"{i}) {element}")
        i = i + 1


def main(ast, pos, matchplayed, crdy, crdr, gls, player, born):
    # Funzione principale per la raccomandazione dei giocatori
    set_recommender(ast, pos, matchplayed, crdy, crdr, gls, player, born)
