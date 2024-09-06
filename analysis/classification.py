import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# Funzione per gestire i valori nulli e convertire la variabile target in categoriale
def preprocess_data(training, target):
    # Gestione dei valori nulli
    training = training.fillna(0)  # Riempie i valori nulli con 0 (o altro valore appropriato)
    target = target.fillna(
        target.mode()[0])  # Riempie i valori nulli nella variabile target con il valore più frequente

    # Conversione della variabile target in categorica se necessario
    if target.dtype != 'object':  # Se la variabile target non è già categoriale
        le = LabelEncoder()
        target = le.fit_transform(target)

    return training, target


# Classificazione KNN
def knn_classification(training, target):
    # Preprocessing dei dati
    training, target = preprocess_data(training, target)

    # Divisione dei dati in train e test
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)

    # Definizione del modello e dei parametri
    knn = KNeighborsClassifier()
    parameters_knn = {
        'n_neighbors': [1, 10, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'manhattan']
    }

    # Grid Search per trovare i migliori parametri
    grid_search_knn = GridSearchCV(
        estimator=knn,
        param_grid=parameters_knn,
        scoring='accuracy',
        n_jobs=-1,
        cv=5
    )
    knn_1 = grid_search_knn.fit(x_train, y_train)
    y_pred_knn1 = knn_1.predict(x_test)
    print("Migliori parametri (KNN):", grid_search_knn.get_params_)

    # Applicazione del miglior modello
    knn = KNeighborsClassifier(**grid_search_knn.best_params_)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    # Risultati
    print("Accuratezza (KNN):", metrics.accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    return knn


# Classificazione Gaussian Naive Bayes
def gaussian_nb_classification(training, target):
    # Preprocessing dei dati
    training, target = preprocess_data(training, target)

    # Divisione dei dati in train e test
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)

    # Definizione del modello e dei parametri
    gau = GaussianNB()
    parameters_gau = {'var_smoothing': np.logspace(0, -9, num=100)}

    # Grid Search per trovare i migliori parametri
    grid_search_gau = GridSearchCV(
        estimator=gau,
        param_grid=parameters_gau,
        cv=10,
        verbose=1,
        scoring='accuracy'
    )
    gau_1 = grid_search_gau.fit(x_train, y_train)
    y_pred_gau1 = gau_1.predict(x_test)
    print("Migliori parametri (GaussianNB):", grid_search_gau.best_params_)

    # Applicazione del miglior modello
    gau = GaussianNB(var_smoothing=grid_search_gau.best_params_['var_smoothing'])
    gau.fit(x_train, y_train)
    y_pred_gau = gau.predict(x_test)

    # Risultati
    print("Accuratezza (GaussianNB):", metrics.accuracy_score(y_test, y_pred_gau))
    print(classification_report(y_test, y_pred_gau, zero_division=0))


# Classificazione Random Forest
def random_forest_classification(training, target):
    # Preprocessing dei dati
    training, target = preprocess_data(training, target)

    # Divisione dei dati in train e test
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)

    # Definizione del modello e dei parametri
    rf = RandomForestClassifier()
    parameters_rf = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [10, 15, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # Randomized Search per trovare i migliori parametri
    random_search_rf = RandomizedSearchCV(
        rf,
        param_distributions=parameters_rf,
        n_iter=20,
        cv=5,
        verbose=1,
        n_jobs=-1,
        scoring='accuracy'
    )

    rf1 = random_search_rf.fit(x_train, y_train)
    y_pred_rf1 = rf1.predict(x_test)
    print("Migliori parametri (RandomForest):", random_search_rf.best_params_)

    # Applicazione del miglior modello
    rf = RandomForestClassifier(**random_search_rf.best_params_)
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)

    # Risultati
    print("Accuratezza (RandomForest):", metrics.accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    return rf
