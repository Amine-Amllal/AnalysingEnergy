Notebook de préparation des données
===================================

Ce notebook guide à travers le processus complet de préparation des données pour l'analyse énergétique avec les modèles LSTM.

Vue d'ensemble
--------------

Le notebook ``Data preprocessing.ipynb`` couvre toutes les étapes nécessaires pour transformer les données brutes du dataset BanE-16 en données prêtes pour l'entraînement des modèles LSTM.

Étapes principales
------------------

1. **Chargement des données**
2. **Analyse exploratoire des données (EDA)**
3. **Nettoyage et filtrage**
4. **Gestion des valeurs manquantes**
5. **Détection et traitement des outliers**
6. **Normalisation et mise à l'échelle**
7. **Création des séquences temporelles**
8. **Division train/validation/test**

Structure du notebook
---------------------

Introduction et imports
~~~~~~~~~~~~~~~~~~~~~~~

Le notebook commence par l'importation des bibliothèques nécessaires et la configuration de l'environnement :

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore')

Chargement et exploration initiale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Chargement des données brutes :**

.. code-block:: python

    # Chargement du dataset principal
    data = pd.read_csv('Data/data.csv', parse_dates=['date'])
    print(f"Forme du dataset : {data.shape}")
    print(f"Période couverte : {data['date'].min()} à {data['date'].max()}")

**Inspection de la structure :**

.. code-block:: python

    # Affichage des informations générales
    data.info()
    
    # Statistiques descriptives
    data.describe()
    
    # Vérification des valeurs manquantes
    missing_values = data.isnull().sum()
    print("Valeurs manquantes par colonne :")
    print(missing_values[missing_values > 0])

Analyse exploratoire des données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Visualisation des tendances temporelles :**

.. code-block:: python

    # Graphique de la production énergétique dans le temps
    plt.figure(figsize=(15, 8))
    plt.plot(data['date'], data['max_generation(mw)'], alpha=0.7)
    plt.title('Évolution de la Production Énergétique')
    plt.xlabel('Date')
    plt.ylabel('Production (MW)')
    plt.grid(True, alpha=0.3)
    plt.show()

**Analyse des corrélations :**

.. code-block:: python

    # Matrice de corrélation
    correlation_matrix = data.select_dtypes(include=[np.number]).corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de Corrélation des Variables')
    plt.tight_layout()
    plt.show()

Nettoyage des données
~~~~~~~~~~~~~~~~~~~~

**Gestion des valeurs manquantes :**

.. code-block:: python

    # Stratégies de traitement selon le type de variable
    def handle_missing_values(df):
        df_clean = df.copy()
        
        # Variables météorologiques : interpolation linéaire
        weather_cols = ['temp2_max(c)', 'temp2_min(c)', 'temp2_ave(c)',
                       'wind_speed50_max(ms)', 'wind_speed50_min(ms)', 'wind_speed50_ave(ms)',
                       'suface_pressure(pa)', 'prectotcorr']
        
        for col in weather_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].interpolate(method='linear')
        
        # Variables énergétiques : forward fill puis backward fill
        energy_cols = ['max_generation(mw)', 'total_demand(mw)']
        for col in energy_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        return df_clean

**Détection des outliers :**

.. code-block:: python

    def detect_outliers(df, columns, threshold=3):
        """Détecte les outliers en utilisant la méthode z-score."""
        outliers_dict = {}
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
                outliers_dict[col] = outliers.index.tolist()
                
                print(f"{col}: {len(outliers)} outliers détectés")
        
        return outliers_dict

**Traitement des outliers :**

.. code-block:: python

    def treat_outliers(df, method='clip', percentile=99):
        """Traite les outliers par clipping ou suppression."""
        df_treated = df.copy()
        
        numeric_cols = df_treated.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'clip':
                # Clipping aux percentiles 1% et 99%
                lower_bound = df_treated[col].quantile(0.01)
                upper_bound = df_treated[col].quantile(0.99)
                df_treated[col] = df_treated[col].clip(lower_bound, upper_bound)
            
            elif method == 'remove':
                # Suppression des outliers
                Q1 = df_treated[col].quantile(0.25)
                Q3 = df_treated[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_treated = df_treated[
                    (df_treated[col] >= lower_bound) & 
                    (df_treated[col] <= upper_bound)
                ]
        
        return df_treated

Normalisation et mise à l'échelle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Préparation des scalers :**

.. code-block:: python

    # Séparation des features et target
    feature_columns = ['temp2_max(c)', 'temp2_min(c)', 'temp2_ave(c)',
                      'suface_pressure(pa)', 'wind_speed50_max(ms)', 
                      'wind_speed50_min(ms)', 'wind_speed50_ave(ms)',
                      'prectotcorr', 'total_demand(mw)']
    
    target_column = 'max_generation(mw)'
    
    X = data_clean[feature_columns].values
    y = data_clean[target_column].values.reshape(-1, 1)

**Application des scalers :**

.. code-block:: python

    # Normalisation des features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Normalisation du target
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    print(f"Forme des features normalisées : {X_scaled.shape}")
    print(f"Forme du target normalisé : {y_scaled.shape}")

Création des séquences temporelles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fonction de création de séquences :**

.. code-block:: python

    def create_sequences(X, y, time_steps=60):
        """Crée des séquences temporelles pour l'entraînement LSTM."""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - time_steps):
            X_seq.append(X[i:(i + time_steps)])
            y_seq.append(y[i + time_steps])
        
        return np.array(X_seq), np.array(y_seq)

**Application :**

.. code-block:: python

    # Paramètres de séquence
    TIME_STEPS = 60  # 60 jours d'historique
    
    # Création des séquences
    X_sequences, y_sequences = create_sequences(X_scaled, y_scaled, TIME_STEPS)
    
    print(f"Forme des séquences X : {X_sequences.shape}")
    print(f"Forme des séquences y : {y_sequences.shape}")

Division des données
~~~~~~~~~~~~~~~~~~~

**Split train/validation/test :**

.. code-block:: python

    # Division temporelle pour respecter l'ordre chronologique
    train_size = int(0.7 * len(X_sequences))
    val_size = int(0.2 * len(X_sequences))
    
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    
    X_val = X_sequences[train_size:train_size + val_size]
    y_val = y_sequences[train_size:train_size + val_size]
    
    X_test = X_sequences[train_size + val_size:]
    y_test = y_sequences[train_size + val_size:]
    
    print(f"Train: {X_train.shape[0]} échantillons")
    print(f"Validation: {X_val.shape[0]} échantillons")
    print(f"Test: {X_test.shape[0]} échantillons")

Sauvegarde des données préparées
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Export des données traitées :**

.. code-block:: python

    # Sauvegarde des données
    np.save('Data/X_train.npy', X_train)
    np.save('Data/y_train.npy', y_train)
    np.save('Data/X_val.npy', X_val)
    np.save('Data/y_val.npy', y_val)
    np.save('Data/X_test.npy', X_test)
    np.save('Data/y_test.npy', y_test)

**Sauvegarde des scalers :**

.. code-block:: python

    import joblib
    
    # Sauvegarde des scalers pour utilisation future
    joblib.dump(scaler_X, 'Notebooks/scalers/X_train_scaler.pkl')
    joblib.dump(scaler_y, 'Notebooks/scalers/y_train_scaler.pkl')

Validation de la préparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Vérifications finales :**

.. code-block:: python

    # Vérification de la qualité des données
    def validate_preprocessing(X_train, y_train, X_val, y_val, X_test, y_test):
        """Valide la qualité de la préparation des données."""
        
        # Vérification des formes
        assert X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:]
        assert y_train.shape[1:] == y_val.shape[1:] == y_test.shape[1:]
        
        # Vérification de l'absence de NaN
        assert not np.isnan(X_train).any()
        assert not np.isnan(y_train).any()
        assert not np.isnan(X_val).any()
        assert not np.isnan(y_val).any()
        assert not np.isnan(X_test).any()
        assert not np.isnan(y_test).any()
        
        # Vérification des distributions
        print("Statistiques des données d'entraînement :")
        print(f"X_train - Min: {X_train.min():.3f}, Max: {X_train.max():.3f}")
        print(f"y_train - Min: {y_train.min():.3f}, Max: {y_train.max():.3f}")
        
        print("✅ Validation réussie !")
    
    validate_preprocessing(X_train, y_train, X_val, y_val, X_test, y_test)

Résumé des transformations
~~~~~~~~~~~~~~~~~~~~~~~~~

**Récapitulatif des étapes appliquées :**

.. code-block:: python

    # Affichage du résumé de préparation
    summary = {
        'Dataset original': data.shape,
        'Après nettoyage': data_clean.shape,
        'Features sélectionnées': len(feature_columns),
        'Séquences créées': X_sequences.shape[0],
        'Longueur séquence': TIME_STEPS,
        'Train/Val/Test': f"{len(X_train)}/{len(X_val)}/{len(X_test)}"
    }
    
    print("📊 Résumé de la préparation des données :")
    for key, value in summary.items():
        print(f"  {key}: {value}")

Utilisation pratique
-------------------

**Exécution du notebook :**

1. Assurez-vous que le fichier ``Data/data.csv`` est présent
2. Exécutez toutes les cellules dans l'ordre
3. Vérifiez la création des fichiers de sortie dans ``Data/`` et ``Notebooks/scalers/``

**Personnalisation :**

- Modifiez ``TIME_STEPS`` pour changer la longueur des séquences
- Ajustez les méthodes de traitement des outliers selon vos besoins
- Adaptez les ratios de division train/val/test

**Fichiers générés :**

- ``X_train.npy``, ``y_train.npy`` : Données d'entraînement
- ``X_val.npy``, ``y_val.npy`` : Données de validation
- ``X_test.npy``, ``y_test.npy`` : Données de test
- ``X_train_scaler.pkl``, ``y_train_scaler.pkl`` : Scalers sauvegardés

Prochaines étapes
----------------

Après avoir préparé les données avec ce notebook, vous pouvez :

1. Consulter :doc:`lstm_training` pour l'entraînement des modèles
2. Explorer :doc:`../model_evaluation` pour évaluer les performances
3. Utiliser :doc:`prediction` pour faire des prédictions

Troubleshooting
---------------

**Problèmes courants :**

- **Erreur de mémoire** : Réduisez ``TIME_STEPS`` ou le batch size
- **Valeurs manquantes** : Vérifiez la qualité du dataset source
- **Performance lente** : Utilisez un sous-échantillon pour les tests

Pour plus d'aide, consultez la section :doc:`../troubleshooting`.
