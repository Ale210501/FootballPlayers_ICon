from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from classification import *
from preprocessing import *

# Carica i dati dal file CSV
data = pd.read_csv('../dataset/top5-players.csv')

# Seleziona le colonne desiderate
data = data.iloc[:, [1, 3, 7, 11, 12, 13, 18, 19]]

# Crea una copia del DataFrame data e la assegna a player
player = data.copy()

# Pulisci i dati rimuovendo valori nulli, resettando gli indici e sistemando le posizioni multiple
player = clean_dataframe(player)

# Converte la colonna 'Born' in tipo numerico e arrotonda i valori nulli a 0
player['Born'] = pd.to_numeric(player['Born'], errors='coerce').fillna(0).astype(int)

# Converte la colonna '90s' in numerico, arrotonda e converte in intero
player['90s'] = pd.to_numeric(player['90s'], errors='coerce').fillna(0)
player['90s'] = player['90s'].round(0).astype(int)

# Converte le colonne 'Gls', 'Ast', 'CrdY', 'CrdR' in numerico con gestione degli errori
player['Gls'] = pd.to_numeric(player['Gls'], errors='coerce')
player['Ast'] = pd.to_numeric(player['Ast'], errors='coerce')
player['CrdY'] = pd.to_numeric(player['CrdY'], errors='coerce')
player['CrdR'] = pd.to_numeric(player['CrdR'], errors='coerce')

# Creazione del dizionario di frequenza per la colonna 'Pos' e selezione della posizione più frequente
player.apply(lambda row: creation_frequency_dictionary(row, dictionary_frequency_pos, 'Pos'), axis=1)
player['Pos_format'] = player.apply(lambda row: set_columns_player(row, dictionary_frequency_pos, 'Pos'), axis=1)

# Ordina i dati per 'Pos_format'
player.sort_values(['Pos_format'], ascending=True, inplace=True)

# Creazione del dizionario di conversione per i nomi dei giocatori
conversion_string(player['Player'])

# Salva il dizionario di conversione in un file CSV
conversionDataset = pd.DataFrame()
conversionDataset['Chiave'] = conversion_dictionary.keys()
conversionDataset['Valore'] = conversion_dictionary.values()
conversionDataset.to_csv("../dataset/dizionario.csv", index=False)

# Conversione dei dati categorici 'Player' in numerici utilizzando il dizionario creato
player['Player_format'] = player.apply(lambda row: convert_by_column(row, 'Player'), axis=1)

# Conversione della colonna 'Pos_format' in valori numerici utilizzando il dizionario pos_dictionary
player['Pos_format'] = player.apply(lambda row: convert_by_pos(row, 'Pos_format'), axis=1)

# Rimuove le colonne non necessarie ('Player' e 'Pos') e salva il DataFrame preprocessato
player_pp = player.drop(columns=['Player', 'Pos'])
player_pp.to_csv("../dataset/players_preprocessato.csv", index=False)

# INIZIO CLASSIFICAZIONE

# Definisce il target come la colonna 'Pos_format' e il set di addestramento come le altre colonne
target = player_pp['Pos_format']
training = player_pp.drop(columns=['Pos_format'])

# Preprocessing dei dati e della variabile target
training, target = preprocess_data(training, target)

# Standardizzazione dei dati numerici
scaler = preprocessing.StandardScaler()
numeric_columns = training.select_dtypes(include=['number']).columns
training[numeric_columns] = scaler.fit_transform(training[numeric_columns])

# Valutazione dei modelli con i parametri
print("K-Nearest Neighbors Results:")
print(evaluate_model_with_cv(KNeighborsClassifier(), knn_params, training, target))
print()  # Riga vuota per separare i risultati dei modelli

print("Gaussian Naive Bayes Results:")
print(evaluate_model_with_cv(GaussianNB(), gau_params, training, target))
print()  # Riga vuota per separare i risultati dei modelli

print("Random Forest Results:")
print(evaluate_model_with_cv(RandomForestClassifier(), rf_params, training, target))

'''
# Funzione per calcolare la percentuale di valori mancanti
def missing_percentage(df):
    return df.isnull().mean() * 100

# Funzione per calcolare il range e la media delle variabili numeriche
def range_and_mean(df):
    # Calcola il range (max - min) per ogni colonna numerica
    ranges = df.max() - df.min()
    # Calcola la media di ogni colonna numerica
    means = df.mean()
    return ranges, means

# Percentuale di valori mancanti per ogni colonna
missing_percentages = missing_percentage(player_pp)
print("Percentuale di valori mancanti per ogni colonna:")
print(missing_percentages)
print()  # Riga vuota per separare i risultati

# Calcola il range e la media delle variabili numeriche prima della normalizzazione
ranges_before, means_before = range_and_mean(player_pp[numeric_columns])
print("Range delle variabili numeriche prima della normalizzazione:")
print(ranges_before)
print("Media delle variabili numeriche prima della normalizzazione:")
print(means_before)
print()  # Riga vuota per separare i risultati

# Standardizzazione dei dati numerici
scaler = preprocessing.StandardScaler()
numeric_columns = training.select_dtypes(include=['number']).columns
training[numeric_columns] = scaler.fit_transform(training[numeric_columns])

# Calcola il range e la media delle variabili numeriche dopo la normalizzazione
ranges_after, means_after = range_and_mean(training[numeric_columns])
print("Range delle variabili numeriche dopo la normalizzazione:")
print(ranges_after)
print("Media delle variabili numeriche dopo la normalizzazione:")
print(means_after)
print()  # Riga vuota per separare i risultati
'''
