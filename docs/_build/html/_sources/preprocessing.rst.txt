Prétraitement des Données
========================

Cette section détaille les étapes de prétraitement appliquées au dataset BanE-16 pour optimiser les performances des modèles LSTM de prédiction énergétique.

Vue d'ensemble du prétraitement
-------------------------------

Le prétraitement des données est une étape critique qui transforme les données brutes en format optimal pour l'entraînement des modèles LSTM. Notre pipeline comprend :

* **Nettoyage des données** et gestion des valeurs manquantes
* **Normalisation et standardisation** des variables
* **Création de features** temporelles et dérivées
* **Division temporelle** des données (train/validation/test)
* **Création de séquences** pour les modèles LSTM

Architecture du pipeline
------------------------

.. code-block:: python

   class DataPreprocessor:
       def __init__(self, sequence_length=60, target_variable='max_generation(mw)'):
           self.sequence_length = sequence_length
           self.target_variable = target_variable
           self.scalers = {}
           self.feature_columns = None
           
       def fit_transform(self, data):
           """Pipeline complet de prétraitement"""
           # 1. Nettoyage des données
           data_cleaned = self.clean_data(data)
           
           # 2. Feature engineering
           data_features = self.create_features(data_cleaned)
           
           # 3. Normalisation
           data_scaled = self.scale_features(data_features)
           
           # 4. Création des séquences
           X, y = self.create_sequences(data_scaled)
           
           return X, y

Nettoyage des données
---------------------

Gestion des valeurs manquantes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Les valeurs manquantes sont traitées selon plusieurs stratégies :

.. code-block:: python

   def clean_data(self, data):
       """Nettoyage complet du dataset"""
       
       # 1. Détection des valeurs manquantes
       missing_data = data.isnull().sum()
       print("Valeurs manquantes par colonne:")
       print(missing_data[missing_data > 0])
       
       # 2. Imputation par interpolation linéaire
       numeric_columns = data.select_dtypes(include=[np.number]).columns
       for col in numeric_columns:
           data[col] = data[col].interpolate(method='linear')
           
       # 3. Imputation par forward fill pour les valeurs en début/fin
       data = data.fillna(method='ffill').fillna(method='bfill')
       
       # 4. Suppression des lignes avec trop de valeurs manquantes
       threshold = len(data.columns) * 0.5  # 50% de valeurs manquantes max
       data = data.dropna(thresh=threshold)
       
       return data

Détection et traitement des outliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def detect_and_treat_outliers(self, data, method='iqr'):
       """Détection et traitement des valeurs aberrantes"""
       
       if method == 'iqr':
           for column in data.select_dtypes(include=[np.number]).columns:
               Q1 = data[column].quantile(0.25)
               Q3 = data[column].quantile(0.75)
               IQR = Q3 - Q1
               
               # Définition des limites
               lower_bound = Q1 - 1.5 * IQR
               upper_bound = Q3 + 1.5 * IQR
               
               # Écrêtage des valeurs aberrantes
               data[column] = data[column].clip(lower_bound, upper_bound)
               
       return data

Feature Engineering
-------------------

Création de variables temporelles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Les variables temporelles capturent les patterns cycliques et saisonniers :

.. code-block:: python

   def create_temporal_features(self, data):
       """Création de features temporelles"""
       
       # Conversion en datetime si nécessaire
       if 'date' in data.columns:
           data['date'] = pd.to_datetime(data['date'])
           data = data.set_index('date')
       
       # Features cycliques
       data['day_of_year'] = data.index.dayofyear
       data['day_of_week'] = data.index.dayofweek
       data['month'] = data.index.month
       data['quarter'] = data.index.quarter
       
       # Encoding cyclique pour préserver la continuité
       data['day_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
       data['day_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
       data['week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
       data['week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
       
       return data

Variables dérivées météorologiques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_weather_features(self, data):
       """Création de features météorologiques dérivées"""
       
       # Différences de température
       data['temp_range'] = data['max_temperature'] - data['min_temperature']
       data['temp_variance'] = data['mean_temperature'].rolling(7).std()
       
       # Features de vent
       data['wind_range'] = data['max_windspeed'] - data['min_windspeed']
       data['wind_gust_factor'] = data['max_windspeed'] / (data['mean_windspeed'] + 1e-8)
       
       # Moyennes mobiles
       windows = [3, 7, 14, 30]
       for window in windows:
           data[f'temp_ma_{window}'] = data['mean_temperature'].rolling(window).mean()
           data[f'wind_ma_{window}'] = data['mean_windspeed'].rolling(window).mean()
           data[f'generation_ma_{window}'] = data['max_generation(mw)'].rolling(window).mean()
       
       # Features de lag
       lags = [1, 2, 3, 7, 14]
       for lag in lags:
           data[f'generation_lag_{lag}'] = data['max_generation(mw)'].shift(lag)
           data[f'wind_lag_{lag}'] = data['mean_windspeed'].shift(lag)
       
       return data

Indices composites
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_composite_indices(self, data):
       """Création d'indices composites"""
       
       # Indice de conditions météorologiques favorables
       data['weather_index'] = (
           0.6 * (data['mean_windspeed'] / data['mean_windspeed'].max()) +
           0.3 * (data['mean_temperature'] / data['mean_temperature'].max()) +
           0.1 * (1 - data['total_precipitation'] / data['total_precipitation'].max())
       )
       
       # Indice de potentiel énergétique
       data['energy_potential'] = (
           data['mean_windspeed'] ** 3  # Loi cubique pour l'éolien
       ) * (1 + 0.1 * data['mean_temperature'])
       
       return data

Normalisation et standardisation
--------------------------------

Stratégies de normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Différentes méthodes de normalisation sont appliquées selon le type de variable :

.. code-block:: python

   from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
   
   def scale_features(self, data):
       """Normalisation des features"""
       
       # Séparation des types de variables
       weather_vars = ['min_temperature', 'mean_temperature', 'max_temperature',
                      'min_windspeed', 'mean_windspeed', 'max_windspeed',
                      'total_precipitation', 'surface_pressure', 'mean_relative_humidity']
       
       derived_vars = [col for col in data.columns if any(x in col for x in ['_ma_', '_lag_', 'range', 'index'])]
       
       target_var = ['max_generation(mw)']
       
       # StandardScaler pour les variables météorologiques
       self.scalers['weather'] = StandardScaler()
       data[weather_vars] = self.scalers['weather'].fit_transform(data[weather_vars])
       
       # RobustScaler pour les variables dérivées (plus résistant aux outliers)
       if derived_vars:
           self.scalers['derived'] = RobustScaler()
           data[derived_vars] = self.scalers['derived'].fit_transform(data[derived_vars])
       
       # MinMaxScaler pour la variable cible (meilleure pour LSTM)
       self.scalers['target'] = MinMaxScaler()
       data[target_var] = self.scalers['target'].fit_transform(data[target_var])
       
       return data

Sauvegarde des scalers
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pickle
   import os
   
   def save_scalers(self, save_dir='scalers/'):
       """Sauvegarde des scalers pour utilisation future"""
       
       os.makedirs(save_dir, exist_ok=True)
       
       for scaler_name, scaler in self.scalers.items():
           filename = os.path.join(save_dir, f'{scaler_name}_scaler.pkl')
           with open(filename, 'wb') as f:
               pickle.dump(scaler, f)
       
       print(f"Scalers sauvegardés dans {save_dir}")

Création de séquences pour LSTM
-------------------------------

Structure des séquences
~~~~~~~~~~~~~~~~~~~~~~~

Les modèles LSTM nécessitent des séquences temporelles en entrée :

.. code-block:: python

   def create_sequences(self, data, sequence_length=60):
       """Création des séquences pour LSTM"""
       
       # Préparation des données
       feature_columns = [col for col in data.columns if col != self.target_variable]
       
       X_data = data[feature_columns].values
       y_data = data[self.target_variable].values
       
       X_sequences = []
       y_sequences = []
       
       # Création des séquences glissantes
       for i in range(sequence_length, len(data)):
           # Séquence d'entrée (60 jours précédents)
           X_sequences.append(X_data[i-sequence_length:i])
           # Valeur cible (jour suivant)
           y_sequences.append(y_data[i])
       
       return np.array(X_sequences), np.array(y_sequences)

Optimisation de la longueur des séquences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def optimize_sequence_length(self, data, lengths_to_test=range(30, 91, 10)):
       """Optimisation de la longueur des séquences"""
       
       from sklearn.model_selection import TimeSeriesSplit
       from tensorflow.keras.models import Sequential
       from tensorflow.keras.layers import LSTM, Dense
       
       results = {}
       
       for length in lengths_to_test:
           print(f"Test avec longueur de séquence: {length}")
           
           # Création des séquences
           X, y = self.create_sequences(data, sequence_length=length)
           
           # Division train/validation
           tscv = TimeSeriesSplit(n_splits=3)
           scores = []
           
           for train_idx, val_idx in tscv.split(X):
               X_train, X_val = X[train_idx], X[val_idx]
               y_train, y_val = y[train_idx], y[val_idx]
               
               # Modèle simple pour test
               model = Sequential([
                   LSTM(50, input_shape=(length, X.shape[2])),
                   Dense(1)
               ])
               model.compile(optimizer='adam', loss='mse')
               
               # Entraînement
               model.fit(X_train, y_train, epochs=10, verbose=0)
               
               # Évaluation
               score = model.evaluate(X_val, y_val, verbose=0)
               scores.append(score)
           
           results[length] = np.mean(scores)
       
       # Meilleure longueur
       best_length = min(results, key=results.get)
       print(f"Longueur optimale: {best_length}")
       
       return best_length, results

Division des données
--------------------

Stratégie de division temporelle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour les séries temporelles, la division doit respecter l'ordre chronologique :

.. code-block:: python

   def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
       """Division chronologique des données"""
       
       total_samples = len(X)
       train_size = int(total_samples * train_ratio)
       val_size = int(total_samples * val_ratio)
       
       # Division séquentielle
       X_train = X[:train_size]
       y_train = y[:train_size]
       
       X_val = X[train_size:train_size + val_size]
       y_val = y[train_size:train_size + val_size]
       
       X_test = X[train_size + val_size:]
       y_test = y[train_size + val_size:]
       
       print(f"Données d'entraînement: {len(X_train)} échantillons")
       print(f"Données de validation: {len(X_val)} échantillons")
       print(f"Données de test: {len(X_test)} échantillons")
       
       return (X_train, y_train), (X_val, y_val), (X_test, y_test)

Validation croisée temporelle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def time_series_cross_validation(self, X, y, n_splits=5):
       """Validation croisée adaptée aux séries temporelles"""
       
       from sklearn.model_selection import TimeSeriesSplit
       
       tscv = TimeSeriesSplit(n_splits=n_splits)
       cv_scores = []
       
       for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
           print(f"Fold {i+1}/{n_splits}")
           
           X_train_cv, X_val_cv = X[train_idx], X[val_idx]
           y_train_cv, y_val_cv = y[train_idx], y[val_idx]
           
           # Ici vous pouvez insérer votre modèle
           # score = train_and_evaluate_model(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
           # cv_scores.append(score)
       
       return cv_scores

Pipeline complet
----------------

Exemple d'utilisation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Exemple d'utilisation complète du preprocessor
   
   # 1. Chargement des données
   data = pd.read_csv('Data/data.csv')
   
   # 2. Initialisation du preprocessor
   preprocessor = DataPreprocessor(sequence_length=60, target_variable='max_generation(mw)')
   
   # 3. Prétraitement complet
   X, y = preprocessor.fit_transform(data)
   
   # 4. Division des données
   (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(X, y)
   
   # 5. Sauvegarde des scalers
   preprocessor.save_scalers('Notebooks/scalers/')
   
   print(f"Données préparées:")
   print(f"- Shape X_train: {X_train.shape}")
   print(f"- Shape y_train: {y_train.shape}")
   print(f"- Features utilisées: {len(preprocessor.feature_columns)}")

Métriques de qualité
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def evaluate_preprocessing_quality(self, original_data, preprocessed_data):
       """Évaluation de la qualité du prétraitement"""
       
       quality_metrics = {}
       
       # 1. Taux de valeurs manquantes
       missing_before = original_data.isnull().sum().sum()
       missing_after = preprocessed_data.isnull().sum().sum()
       quality_metrics['missing_reduction'] = 1 - (missing_after / missing_before)
       
       # 2. Conservation de l'information
       correlations_before = original_data.corr().abs().mean().mean()
       correlations_after = preprocessed_data.corr().abs().mean().mean()
       quality_metrics['correlation_preservation'] = correlations_after / correlations_before
       
       # 3. Stabilité des distributions
       from scipy.stats import ks_2samp
       ks_stats = []
       common_columns = set(original_data.columns) & set(preprocessed_data.columns)
       
       for col in common_columns:
           if original_data[col].dtype in ['int64', 'float64']:
               ks_stat, _ = ks_2samp(original_data[col].dropna(), 
                                   preprocessed_data[col].dropna())
               ks_stats.append(ks_stat)
       
       quality_metrics['distribution_stability'] = 1 - np.mean(ks_stats)
       
       return quality_metrics

Bonnes pratiques
----------------

Recommandations générales
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Ordre des opérations** : Respecter l'ordre logique (nettoyage → feature engineering → normalisation)
2. **Validation** : Toujours valider les transformations sur un échantillon
3. **Reproductibilité** : Fixer les seeds aléatoires et sauvegarder les paramètres
4. **Documentation** : Documenter chaque transformation appliquée

Éviter les fuites de données
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ❌ Mauvais : normalisation avant division
   data_scaled = scaler.fit_transform(data)
   X_train, X_test = train_test_split(data_scaled)
   
   # ✅ Bon : normalisation après division
   X_train, X_test = train_test_split(data)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)  # Pas de fit !

Optimisation des performances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Vectorisation** : Utiliser NumPy et Pandas pour les opérations
2. **Parallélisation** : Traitement en parallèle pour les gros datasets
3. **Chunking** : Traitement par chunks pour les datasets trop volumineux
4. **Caching** : Mise en cache des transformations coûteuses

Prochaines étapes
-----------------

Après le prétraitement, les données sont prêtes pour :

* :doc:`lstm_models` - Entraînement des modèles LSTM
* :doc:`hyperparameter_optimization` - Optimisation des hyperparamètres
* :doc:`model_evaluation` - Évaluation des performances
