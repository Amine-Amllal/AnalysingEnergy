Démarrage rapide
================

Ce guide vous permettra de démarrer rapidement avec le projet Analysing Green Energy et d'obtenir vos premières prédictions en quelques minutes.

Vue d'ensemble
--------------

Le projet Analysing Green Energy utilise des réseaux de neurones LSTM pour prédire la production d'énergie verte à partir de données météorologiques et de demande énergétique.

Étapes rapides
--------------

1. **Installation** (5 minutes)

.. code-block:: bash

    git clone https://github.com/votre-username/AnalysingEnergy.git
    cd AnalysingEnergy
    pip install -r requirements.txt

2. **Exploration des données** (10 minutes)

Ouvrez le notebook ``Notebooks/Data preprocessing.ipynb`` pour comprendre les données :

.. code-block:: python

    import pandas as pd
    
    # Charger les données
    data = pd.read_csv('Data/data.csv', index_col='date', parse_dates=True)
    print(data.head())
    print(data.describe())

3. **Lancer l'interface** (2 minutes)

.. code-block:: bash

    streamlit run interface/app.py

L'interface s'ouvrira automatiquement dans votre navigateur à l'adresse ``http://localhost:8501``.

4. **Utiliser un modèle pré-entraîné** (5 minutes)

Si vous avez des modèles pré-entraînés dans ``Notebooks/models/`` :

.. code-block:: python

    import joblib
    from tensorflow.keras.models import load_model
    
    # Charger un modèle
    model = load_model('Notebooks/models/final_model 291.19.h5')
    scaler = joblib.load('Notebooks/scalers/total_demand(mw)_scaler.pkl')
    
    # Faire une prédiction
    # (voir les notebooks pour des exemples complets)

Cas d'usage principaux
----------------------

Analyse exploratoire des données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le notebook ``Data preprocessing.ipynb`` contient :

- Description détaillée du dataset BanE-16
- Analyse de la distribution des variables
- Visualisations des tendances temporelles
- Détection des valeurs aberrantes

.. code-block:: python

    # Exemple d'analyse rapide
    import matplotlib.pyplot as plt
    
    # Visualiser la production vs demande
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['max_generation(mw)'], label='Production')
    plt.plot(data.index, data['total_demand(mw)'], label='Demande')
    plt.legend()
    plt.title('Production vs Demande d\'énergie')
    plt.show()

Entraînement d'un modèle LSTM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le notebook ``LSTM Generation.ipynb`` montre comment :

1. Préparer les données pour le LSTM
2. Construire l'architecture du modèle
3. Optimiser les hyperparamètres avec Optuna
4. Évaluer les performances

.. code-block:: python

    # Architecture LSTM basique
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(74, activation='relu', return_sequences=True, input_shape=(1, 9)),
        LSTM(69, activation='relu', return_sequences=False),
        Dropout(0.19),
        Dense(50, activation='relu'),
        Dropout(0.19),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')

Prédictions futures
~~~~~~~~~~~~~~~~~~

Le notebook ``Predicting_next_365days.ipynb`` démontre :

- Chargement des modèles entraînés
- Prédiction pour les 365 prochains jours
- Sauvegarde des résultats

Interface interactive
~~~~~~~~~~~~~~~~~~~~

L'application Streamlit (``interface/app.py``) offre :

- Visualisation en temps réel
- Comparaison production/consommation
- Interface conviviale pour les non-techniciens

Données disponibles
------------------

Le dataset BanE-16 inclut les variables suivantes :

.. list-table:: Variables du dataset
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``temp2_max(c)``
     - Température maximale (°C)
   * - ``temp2_min(c)``
     - Température minimale (°C)
   * - ``temp2_ave(c)``
     - Température moyenne (°C)
   * - ``suface_pressure(pa)``
     - Pression atmosphérique (Pa)
   * - ``wind_speed50_max(m/s)``
     - Vitesse du vent maximale à 50m (m/s)
   * - ``wind_speed50_min(m/s)``
     - Vitesse du vent minimale à 50m (m/s)
   * - ``wind_speed50_ave(m/s)``
     - Vitesse du vent moyenne à 50m (m/s)
   * - ``prectotcorr``
     - Précipitations corrigées
   * - ``total_demand(mw)``
     - Demande totale d'énergie (MW)
   * - ``max_generation(mw)``
     - Production maximale d'énergie (MW) - **Variable cible**

Résultats typiques
------------------

Avec la configuration optimale, vous devriez obtenir :

- **RMSE** : ~291 MW pour la prédiction de génération maximale
- **Temps d'entraînement** : 10-15 minutes sur CPU standard
- **Précision** : 85-90% pour les prédictions à court terme

Modèles disponibles
------------------

Le projet inclut des modèles pré-entraînés pour :

- ``final_model 291.19.h5`` - Modèle principal optimisé
- ``temp2_ave(c)_LSTM.h5`` - Prédiction de température
- ``wind_speed50_ave(ms)_LSTM.h5`` - Prédiction de vitesse du vent
- ``total_demand(mw)_LSTM.h5`` - Prédiction de demande

Conseils pour débuter
--------------------

1. **Commencez par l'exploration** : Familiarisez-vous avec les données avant de plonger dans la modélisation.

2. **Utilisez les notebooks dans l'ordre** :
   - Data preprocessing
   - LSTM Generation  
   - Predicting_next_365days

3. **Expérimentez avec les hyperparamètres** : Utilisez Optuna pour trouver la meilleure configuration pour votre cas d'usage.

4. **Surveillez la convergence** : Utilisez des callbacks pour arrêter l'entraînement si nécessaire.

5. **Validez vos résultats** : Comparez les prédictions avec des données de test non vues.

Prochaines étapes
-----------------

Après ce démarrage rapide :

1. Consultez :doc:`data_description` pour comprendre les données en détail
2. Explorez :doc:`lstm_models` pour approfondir la modélisation
3. Lisez :doc:`hyperparameter_optimization` pour optimiser vos modèles
4. Visitez :doc:`interface` pour personnaliser l'application

Résolution des problèmes courants
---------------------------------

**Le notebook ne se lance pas**

.. code-block:: bash

    jupyter notebook --generate-config
    jupyter notebook

**Erreur de mémoire avec TensorFlow**

.. code-block:: python

    import tensorflow as tf
    tf.config.experimental.set_memory_growth(
        tf.config.experimental.list_physical_devices('GPU')[0], True
    )

**L'interface Streamlit ne s'ouvre pas**

Vérifiez que le port 8501 n'est pas utilisé :

.. code-block:: bash

    streamlit run interface/app.py --server.port 8502

Support
-------

- Consultez la section :doc:`troubleshooting` pour les problèmes courants
- Visitez la :doc:`faq` pour les questions fréquentes
- Consultez les issues GitHub pour des problèmes spécifiques
