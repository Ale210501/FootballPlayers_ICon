import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Carica il dataset preprocessato
dataset = pd.read_csv('../dataset/players_preprocessato.csv')

# 1. Analisi Esplorativa dei Dati (EDA)

# Statistiche descrittive del dataset
print(dataset.describe())

# Visualizza la matrice di correlazione
plt.figure(figsize=(10, 8))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matrice di Correlazione')
plt.show()

# 2. Distribuzione delle variabili numeriche
# Visualizzazione della distribuzione delle principali variabili numeriche
numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(12, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(dataset[feature], kde=True, bins=20, color='blue')
    plt.title(f'Distribuzione di {feature}')
plt.tight_layout()
plt.show()

# 3. Identificazione di Outliers
# Boxplot per identificare gli outliers
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=dataset[feature], color='green')
    plt.title(f'Outliers di {feature}')
plt.tight_layout()
plt.show()

# 4. Relazioni tra variabili numeriche
# Pairplot per osservare le relazioni tra le variabili numeriche
sns.pairplot(dataset[numerical_features])
plt.suptitle('Relazioni tra variabili numeriche', y=1.02)
plt.show()

# 5. Analisi della variabile target
# Distribuzione della variabile target
target_column = 'Pos_format'
plt.figure(figsize=(8, 6))
sns.countplot(x=dataset[target_column])
plt.title('Distribuzione della variabile target')
plt.show()

# 6. Analisi delle correlazioni tra la variabile target e le altre variabili
# Correlazione tra la variabile target e altre variabili numeriche
target_corr = dataset.corr()[target_column].sort_values(ascending=False)
print("Target_corr:")
print(target_corr)

