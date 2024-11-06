import pandas as pd
from warnings import simplefilter
from sklearn.ensemble import RandomForestClassifier
from analysis.preprocessing import pos_dictionary

# Ignora i futuri avvisi di deprecazione
simplefilter(action='ignore', category=FutureWarning)

# Dizionari per la conversione e la gestione dei dati
player_dic = {}
conversion_dic = {}
pos_dic = {}


# FUNZIONI PER PREDIRE IL RUOLO DEL GIOCATORE

def set_player():
    # Imposta il dizionario dei giocatori usando conversion_dic
    for key, value in conversion_dic.items():
        if key == 'fine_player':
            break
        player_dic[key] = value


def search_format_player(element):
    # Cerca il formato del giocatore nel dizionario di conversione
    if element not in conversion_dic:
        conversion_dic[element] = len(conversion_dic)
    return conversion_dic[element]


def search_format_pos(element):
    # Cerca la posizione del giocatore nel dizionario di posizioni
    return pos_dictionary.get(element)


def set_pos():
    # Imposta il dizionario delle posizioni usando pos_dictionary
    for key, value in conversion_dic.items():
        if key == 'fine_pos':
            break
        pos_dic[key] = value


def set_dictionary(row):
    # Popola conversion_dic con i dati del DataFrame
    conversion_dic[row['Chiave']] = row['Valore']


def random_forest_classification(training, target):
    # Configura il modello di Random Forest con i migliori parametri
    rf_model = RandomForestClassifier(
        criterion='entropy',
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(training, target)
    return rf_model


def predict_pos(player, born, matchplayed, gls, ast, crdy, crdr):
    # Carica i dati dei giocatori e il dizionario di conversione
    players = pd.read_csv('../dataset/players_preprocessato.csv')
    conversion = pd.read_csv('../dataset/dizionario.csv')

    # Crea il dizionario di conversione
    conversion.apply(set_dictionary, axis=1)

    # Prepara i dati per la classificazione
    target = players['Pos_format']
    training = players.drop(columns=['Pos_format'])

    # Ottieni l'ID del giocatore dal nome
    player_format = search_format_player(player)

    # Crea un DataFrame con i dati dell'utente
    row_user = pd.DataFrame([[born, matchplayed, gls, ast, crdy, crdr, player_format]],
                            columns=training.columns)

    # Esegui la classificazione con Random Forest usando i migliori parametri
    rf_model = random_forest_classification(training, target)
    pos_predict = rf_model.predict(row_user)

    # Gestisci le posizioni
    predicted_pos = next((key for key, value in pos_dictionary.items() if value == pos_predict[0]), None)

    if predicted_pos:
        print(f"Il ruolo del giocatore {player} è: {predicted_pos}")
    else:
        print(f"Il ruolo del giocatore {player} non è stato trovato. (Previsione: {pos_predict[0]})")


def main(player, born, matchplayed, gls, ast, crdy, crdr):
    # Funzione principale per predire il ruolo del giocatore
    predict_pos(player, born, matchplayed, gls, ast, crdy, crdr)
