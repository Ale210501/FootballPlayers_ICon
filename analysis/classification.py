import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Funzione per gestire i valori nulli e convertire la variabile target in categoriale
def preprocess_data(training, target):
    training = training.fillna(0)  # Riempie i valori nulli con 0
    target = target.fillna(target.mode()[0])  # Riempie i valori nulli con il valore più frequente
    if target.dtype != 'object':
        le = LabelEncoder()
        target = le.fit_transform(target)
    return training, target


# Funzione per calcolare l'accuratezza media e altre metriche al variare della cross-validation
def evaluate_model_with_cv(model, param_grid, training, target, cv_values=[5, 10, 15]):
    # Dizionario per memorizzare le medie dei risultati al variare del cv e i migliori parametri
    results = {'cv': [], 'mean_accuracy': [], 'mean_precision': [], 'mean_recall': [], 'mean_f1': [], 'roc_auc': [], 'best_params': []}

    for cv in cv_values:
        # Ricerca del miglior modello con GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            n_jobs=-1
        )
        grid_search.fit(training, target)

        # Calcolo delle metriche con cross_val_score sul miglior modello trovato
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(training)

        # Calcolo delle metriche
        accuracy = np.mean(cross_val_score(best_model, training, target, cv=cv, scoring='accuracy'))
        precision = precision_score(target, predictions, average='weighted', zero_division=0)
        recall = recall_score(target, predictions, average='weighted', zero_division=0)
        f1 = f1_score(target, predictions, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(target, best_model.predict_proba(training), multi_class='ovr', average='weighted')

        # Aggiunta dei risultati e dei migliori parametri alla tabella
        results['cv'].append(cv)
        results['mean_accuracy'].append(accuracy)
        results['mean_precision'].append(precision)
        results['mean_recall'].append(recall)
        results['mean_f1'].append(f1)
        results['roc_auc'].append(roc_auc)
        results['best_params'].append(grid_search.best_params_)

        # Stampa i risultati per ogni valore di cv
        print(f"\nCross-validation: {cv}-fold")
        print(f"Migliori parametri: {grid_search.best_params_}")
        print(f"Accuratezza media: {accuracy}")
        print(f"Precisione media: {precision}")
        print(f"Richiamo medio: {recall}")
        print(f"F1-Score medio: {f1}")
        print(f"ROC AUC: {roc_auc}")

    return pd.DataFrame(results)  # Converte i risultati in un DataFrame


# Definizione dei parametri per ogni classificatore
knn_params = {
    'n_neighbors': [5, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'cosine']
}

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

gau_params = {
    'var_smoothing': np.logspace(0, -9, num=10)
}
