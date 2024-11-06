from sklearn import preprocessing
from sklearn.metrics import pairwise_distances  # Importa la funzione per calcolare le distanze

from playerposprediction import *


def reformat_player(row, player):
    for key in player_dic:
        if player_dic[key] == row[player]:
            row[player] = key
            return row[player]


def similarity_with_manhattan(row_a, row_b):
    # Calcola la distanza di Manhattan tra due righe
    distance = pairwise_distances([row_a], [row_b], metric='manhattan')
    return 1 / (1 + distance[0][0])  # Invertiamo la distanza per trattarla come similarità


def set_recommender(ast, pos, matchplayed, crdy, crdr, gls, player, born):
    # Carica i dati dei giocatori
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

    # Normalizza i dati dei giocatori e converte in DataFrame
    cluster_dataset_normalized = preprocessing.normalize(cluster_dataset)
    cluster_dataset_normalized = pd.DataFrame(cluster_dataset_normalized, columns=cluster_dataset.columns)

    # Calcola la similarità tra i dati dell'utente e i dati dei giocatori
    cluster_dataset['similarity'] = cluster_dataset_normalized.apply(
        lambda row: similarity_with_manhattan(row, row_user_normalized[0]), axis=1
    )

    # Ordina i giocatori in base alla similarità
    cluster_dataset.sort_values(['similarity'], ascending=False, inplace=True)

    # Imposta il dizionario dei giocatori
    set_player()

    # Seleziona i primi 10 giocatori più simili
    ten_sim = cluster_dataset.head(10)

    # Riformatta i nomi dei giocatori
    ten_sim = ten_sim.loc[:, ['Player_format', 'similarity']]
    ten_sim['Player_format'] = ten_sim.apply(lambda row: reformat_player(row, 'Player_format'), axis=1)

    # Filtra i giocatori suggeriti che sono già stati selezionati dall'utente
    user_selected_player = player_dic.get(player, None)  # Ottieni il valore del giocatore selezionato
    if user_selected_player is not None:
        ten_sim = ten_sim[ten_sim['Player_format'] != user_selected_player]

    # Stampa i giocatori suggeriti
    print("I giocatori suggeriti in base alle tue preferenze sono:\n")

    for i, element in enumerate(ten_sim['Player_format'], start=1):
        print(f"{i}) {element}")


def main(ast, pos, matchplayed, crdy, crdr, gls, player, born):
    set_recommender(ast, pos, matchplayed, crdy, crdr, gls, player, born)
