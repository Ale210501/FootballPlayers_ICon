import pandas as pd
from warnings import simplefilter
from analysis.preprocessing import pos_dictionary
from analysis.classification import random_forest_classification

# Ignora i futuri avvisi di deprecazione
simplefilter(action='ignore', category=FutureWarning)

# Dizionari per la conversione e la gestione dei dati
player_dic = {}
conversion_dic = {}
pos_dic = {}


# FUNZIONI PER PREDIRE IL RUOLO DEL GIOCATORE

def set_player():
    # Imposta il dizionario dei giocatori usando conversion_dic
    for key in conversion_dic:
        if key != 'fine_player':
            player_dic[key] = conversion_dic[key]
        elif key == 'fine_player':
            return


def search_format_player(element):
    # Cerca il formato del giocatore nel dizionario di conversione
    if element in conversion_dic:
        return conversion_dic[element]
    else:
        new_id = len(conversion_dic)
        conversion_dic[element] = new_id
        return new_id


def search_format_pos(element):
    # Cerca la posizione del giocatore nel dizionario di posizioni
    return pos_dictionary.get(element, None)


def set_pos():
    # Imposta il dizionario delle posizioni usando conversion_dic
    for key in conversion_dic:
        if key != 'fine_pos':
            pos_dic[key] = conversion_dic[key]
        else:
            return


def set_dictionary(row):
    # Popola conversion_dic con i dati del DataFrame
    conversion_dic[row['Chiave']] = row['Valore']


def predict_pos(player, born, matchplayed, gls, ast, crdy, crdr):
    # Carica i dati dei giocatori e il dizionario di conversione
    players = pd.read_csv('../dataset/players_preprocessato.csv')
    conversion = pd.read_csv('../dataset/dizionario.csv')

    # Crea il dizionario di conversione
    conversion.apply(lambda row: set_dictionary(row), axis=1)

    # Prepara i dati per la classificazione
    target = players['Pos_format']
    training = players.drop(columns=['Pos_format'])

    # Ottieni l'ID del giocatore dal nome
    player_format = search_format_player(player)

    # Crea un DataFrame con i dati dell'utente
    row_user = pd.DataFrame([[born, matchplayed, gls, ast, crdy, crdr, player_format]],
                            columns=training.columns)

    # Esegui la classificazione con Random Forest
    rf = random_forest_classification(training, target)
    pos_predict = rf.predict(row_user)

    # Gestisci le posizioni
    predicted_pos = [key for key, value in pos_dictionary.items() if value == pos_predict[0]]

    if predicted_pos:
        print(f"Il ruolo del giocatore {player} è: ", predicted_pos[0])
    else:
        print(f"Il ruolo del giocatore {player} non è stato trovato. (Previsione: {pos_predict[0]})")


def main(player, born, matchplayed, gls, ast, crdy, crdr):
    # Funzione principale per predire il ruolo del giocatore
    predict_pos(player, born, matchplayed, gls, ast, crdy, crdr)
