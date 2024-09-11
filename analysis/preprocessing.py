import pandas as pd

# Dizionari globali per conversioni e frequenze
dictionary_pos_format = {}
dictionary_frequency_pos = {}
conversion_dictionary = {}
player_dictionary = {}
# Dizionario per convertire le posizioni dei giocatori in valori numerici
pos_dictionary = {'gk': 0, 'df': 1, 'mf': 2, 'fw': 3}


def conversion_string(column_name):

    i = 0
    for player in column_name:
        if player not in conversion_dictionary:
            conversion_dictionary[player] = i
            i += 1
    conversion_dictionary["fine_player"] = -1


def creation_frequency_dictionary(row, dictionary, column_name):

    list_ = str(row[column_name]).split(', ')
    for element in list_:
        if element not in dictionary:
            dictionary[element] = 1
        else:
            dictionary[element] += 1


def set_columns_player(row, dictionary, column_name):

    list_ = str(row[column_name]).split(', ')
    pos_frequency = 0
    final_pos = ''
    for player in list_:
        if player in dictionary and dictionary[player] > pos_frequency:
            pos_frequency = dictionary[player]
            final_pos = player
    row[column_name] = final_pos
    return row[column_name]


def convert_by_column(row, column_name):

    if row[column_name] in conversion_dictionary:
        element = row[column_name]
        row[column_name] = conversion_dictionary[element]
    return row[column_name]


def convert_by_pos(row, column_name):

    if row[column_name] in pos_dictionary:
        element = row[column_name]
        row[column_name] = pos_dictionary[element]
    return row[column_name]


def clean_dataframe(player):

    # Elimina le righe con valori nulli
    player = player.dropna()

    # Reset dell'indice
    player = player.reset_index(drop=True)

    # Elimina i valori doppi presenti tra 'Pos' di ogni giocatore e lascia il valore migliore
    player["Pos"] = player["Pos"].replace("MF,FW", 'FW')
    player["Pos"] = player["Pos"].replace("DF,MF", 'DF')
    player["Pos"] = player["Pos"].replace("FW,DF", 'MF')
    player["Pos"] = player["Pos"].replace("FW,MF", 'FW')
    player["Pos"] = player["Pos"].replace("MF,DF", 'DF')
    player["Pos"] = player["Pos"].replace("DF,FW", 'MF')

    # Converte tutto in minuscolo
    player = player.map(lambda x: str(x).lower() if pd.notnull(x) else x)

    return player