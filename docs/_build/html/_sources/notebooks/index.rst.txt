Documentation des Notebooks
===========================

Cette section présente les notebooks Jupyter utilisés dans le projet AnalysingEnergy pour l'analyse des données et le développement des modèles LSTM.

Vue d'ensemble des notebooks
----------------------------

Le projet comprend plusieurs notebooks spécialisés pour différentes étapes du pipeline de machine learning :

* **Data preprocessing.ipynb** - Prétraitement et nettoyage des données
* **LSTM Generation.ipynb** - Développement du modèle LSTM principal  
* **Predicting_next_365days.ipynb** - Prédictions à long terme
* **LSTM complet.ipynb** - Implémentation complète du système
* **LSTM complet interface.ipynb** - Intégration avec l'interface Streamlit

Structure des notebooks
-----------------------

Chaque notebook suit une structure standardisée :

1. **Imports et configuration** des librairies
2. **Chargement des données** et exploration initiale
3. **Prétraitement** adapté à l'objectif
4. **Modélisation** et entraînement
5. **Évaluation** des résultats
6. **Visualisations** et interprétations
7. **Sauvegarde** des modèles et résultats

Data preprocessing.ipynb
------------------------

Objectif
~~~~~~~~

Ce notebook se concentre sur le prétraitement complet du dataset BanE-16 pour optimiser les données d'entrée des modèles LSTM.

Contenu principal
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Structure typique du notebook de prétraitement
   
   # 1. Exploration des données
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Chargement des données
   data = pd.read_csv('../Data/data.csv')
   print(f"Shape du dataset: {data.shape}")
   print(f"Colonnes: {list(data.columns)}")
   
   # 2. Analyse des valeurs manquantes
   missing_data = data.isnull().sum()
   print("Valeurs manquantes par colonne:")
   print(missing_data[missing_data > 0])
   
   # 3. Statistiques descriptives
   print(data.describe())

Étapes de prétraitement
~~~~~~~~~~~~~~~~~~~~~~

1. **Nettoyage des données**
   
   - Détection et traitement des valeurs manquantes
   - Identification des valeurs aberrantes
   - Correction des incohérences temporelles

2. **Feature engineering**
   
   - Création de variables temporelles (jour, mois, saison)
   - Calcul de moyennes mobiles
   - Génération de variables lag

3. **Normalisation**
   
   - Standardisation des variables météorologiques
   - Normalisation de la variable cible
   - Sauvegarde des scalers

4. **Division des données**
   
   - Séparation chronologique train/validation/test
   - Création des séquences pour LSTM

Sorties du notebook
~~~~~~~~~~~~~~~~~~

- Datasets préprocessés (train_data.csv, test_data.csv)
- Scalers sauvegardés (dossier scalers/)
- Graphiques d'analyse exploratoire
- Rapport de qualité des données

LSTM Generation.ipynb
---------------------

Objectif
~~~~~~~~

Développement et entraînement du modèle LSTM principal pour la prédiction de génération d'énergie verte.

Architecture du modèle
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Architecture LSTM optimisée
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense, Dropout
   
   def create_lstm_model(input_shape, units_1=74, units_2=69, dropout_rate=0.1938):
       model = Sequential([
           LSTM(units_1, return_sequences=True, input_shape=input_shape),
           Dropout(dropout_rate),
           LSTM(units_2, return_sequences=False),
           Dropout(dropout_rate),
           Dense(1, activation='linear')
       ])
       
       model.compile(
           optimizer='adam',
           loss='mse',
           metrics=['mae']
       )
       
       return model

Processus d'entraînement
~~~~~~~~~~~~~~~~~~~~~~~

1. **Préparation des données**
   
   - Chargement des données préprocessées
   - Création des séquences temporelles (60 jours)
   - Division train/validation

2. **Configuration du modèle**
   
   - Architecture LSTM à 2 couches
   - Optimisation des hyperparamètres
   - Callbacks pour early stopping

3. **Entraînement**
   
   - Entraînement sur données historiques
   - Validation sur données de test
   - Monitoring des métriques

4. **Évaluation**
   
   - Calcul RMSE, MAE, R²
   - Analyse des résidus
   - Visualisation des prédictions

Résultats obtenus
~~~~~~~~~~~~~~~~

Le modèle final achieve :

- **RMSE** : 291.19 MW
- **MAE** : ~185 MW  
- **R²** : 0.847
- **Temps d'entraînement** : ~45 minutes

Predicting_next_365days.ipynb
-----------------------------

Objectif
~~~~~~~~

Génération de prédictions à long terme (365 jours) pour la planification énergétique.

Méthodologie
~~~~~~~~~~~

.. code-block:: python

   def predict_long_term(model, last_sequence, n_days=365, scaler=None):
       """Prédiction récursive à long terme"""
       
       predictions = []
       current_sequence = last_sequence.copy()
       
       for day in range(n_days):
           # Prédiction du jour suivant
           pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))
           predictions.append(pred[0, 0])
           
           # Mise à jour de la séquence (glissement)
           # Ici on simulerait les nouvelles données météo
           # Pour simplicité, on utilise les dernières valeurs connues
           new_features = current_sequence[-1].copy()
           new_features[-1] = pred[0, 0]  # Actualiser la génération prédite
           
           # Glissement de la fenêtre
           current_sequence = np.vstack([current_sequence[1:], new_features])
       
       return np.array(predictions)

Défis de la prédiction long terme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Accumulation d'erreurs** : Les erreurs se propagent dans les prédictions futures
2. **Incertitude météorologique** : Difficulté à prédire les conditions météo
3. **Changements saisonniers** : Adaptation aux variations saisonnières
4. **Événements exceptionnels** : Gestion des conditions extrêmes

Stratégies d'amélioration
~~~~~~~~~~~~~~~~~~~~~~~~

- **Mise à jour régulière** avec nouvelles données
- **Ensemble de modèles** pour réduire l'incertitude
- **Intégration de prévisions météo** externes
- **Intervalles de confiance** pour quantifier l'incertitude

LSTM complet.ipynb
------------------

Objectif
~~~~~~~~

Implémentation complète du système de prédiction énergétique avec toutes les fonctionnalités.

Fonctionnalités intégrées
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Pipeline complet**
   
   - Prétraitement automatisé
   - Entraînement du modèle
   - Évaluation et validation
   - Sauvegarde des résultats

2. **Optimisation des hyperparamètres**
   
   - Utilisation d'Optuna
   - Optimisation multi-objectif
   - Validation croisée temporelle

3. **Modèles spécialisés**
   
   - Modèles par variable météorologique
   - Ensemble de modèles
   - Modèle de consensus

4. **Interface de test**
   
   - Fonctions de test interactives
   - Visualisations avancées
   - Export des résultats

Structure du notebook
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Structure du notebook complet
   
   # Section 1: Configuration et imports
   import sys
   sys.path.append('../interface/')
   from app import EnergyPredictor
   
   # Section 2: Chargement et préparation des données
   predictor = EnergyPredictor()
   predictor.load_data('../Data/data.csv')
   
   # Section 3: Entraînement des modèles
   predictor.train_models()
   
   # Section 4: Évaluation
   results = predictor.evaluate_models()
   
   # Section 5: Prédictions et visualisations
   predictions = predictor.predict_future(days=30)
   predictor.plot_results()

LSTM complet interface.ipynb
----------------------------

Objectif
~~~~~~~~

Développement et test de l'interface Streamlit pour l'application de prédiction énergétique.

Composants de l'interface
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Interface de chargement**
   
   - Upload de fichiers de données
   - Validation du format
   - Prévisualisation des données

2. **Configuration des modèles**
   
   - Sélection des paramètres
   - Choix des variables d'entrée
   - Configuration de l'entraînement

3. **Visualisations interactives**
   
   - Graphiques temps réel
   - Comparaisons de modèles
   - Métriques de performance

4. **Export des résultats**
   
   - Téléchargement des prédictions
   - Rapports PDF
   - Données pour analyse externe

Tests d'intégration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test de l'interface dans le notebook
   
   # Simulation de l'interaction utilisateur
   test_data = pd.read_csv('../Data/test_data.csv')
   
   # Test des fonctions de l'interface
   from interface.app import EnergyPredictor
   
   app = EnergyPredictor()
   app.load_data_from_dataframe(test_data)
   
   # Test de prédiction
   prediction = app.predict_single_day(
       min_temp=15.2,
       mean_temp=22.1,
       max_temp=28.9,
       wind_speed=12.5,
       precipitation=0.0,
       pressure=1013.25,
       humidity=65.0
   )
   
   print(f"Prédiction: {prediction:.2f} MW")

Bonnes pratiques pour les notebooks
-----------------------------------

Structure et organisation
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Titre et description** clairs en en-tête
2. **Table des matières** pour les notebooks longs
3. **Sections bien délimitées** avec markdown
4. **Documentation** des fonctions et paramètres

Qualité du code
~~~~~~~~~~~~~~

.. code-block:: python

   # Exemple de bonne pratique
   
   def preprocess_weather_data(data, target_col='max_generation(mw)'):
       """
       Prétraite les données météorologiques pour l'entraînement LSTM.
       
       Parameters:
       -----------
       data : pd.DataFrame
           Données brutes du dataset BanE-16
       target_col : str
           Nom de la colonne cible
           
       Returns:
       --------
       X : np.array
           Features préprocessées
       y : np.array
           Variable cible
       """
       
       # Implémentation avec gestion d'erreurs
       try:
           # Prétraitement...
           pass
       except Exception as e:
           print(f"Erreur dans le prétraitement: {e}")
           return None, None

Reproductibilité
~~~~~~~~~~~~~~~

1. **Seeds fixés** pour la reproductibilité
2. **Versions des librairies** documentées
3. **Chemins relatifs** pour la portabilité
4. **Sauvegarde** des résultats intermédiaires

Visualisations
~~~~~~~~~~~~~

.. code-block:: python

   # Exemple de visualisation standardisée
   
   def plot_training_history(history, save_path=None):
       """Graphique standardisé de l'historique d'entraînement"""
       
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
       
       # Loss
       ax1.plot(history.history['loss'], label='Train Loss')
       ax1.plot(history.history['val_loss'], label='Validation Loss')
       ax1.set_title('Évolution de la Loss')
       ax1.set_xlabel('Epoch')
       ax1.set_ylabel('MSE')
       ax1.legend()
       ax1.grid(True)
       
       # Métriques
       ax2.plot(history.history['mae'], label='Train MAE')
       ax2.plot(history.history['val_mae'], label='Validation MAE')
       ax2.set_title('Évolution du MAE')
       ax2.set_xlabel('Epoch')
       ax2.set_ylabel('MAE')
       ax2.legend()
       ax2.grid(True)
       
       plt.tight_layout()
       
       if save_path:
           plt.savefig(save_path, dpi=300, bbox_inches='tight')
       
       plt.show()

Utilisation des notebooks
-------------------------

Ordre d'exécution recommandé
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Data preprocessing.ipynb** - Préparation des données
2. **LSTM Generation.ipynb** - Développement du modèle principal
3. **LSTM complet.ipynb** - Validation et optimisation
4. **Predicting_next_365days.ipynb** - Tests de prédiction long terme
5. **LSTM complet interface.ipynb** - Test de l'interface

Configuration requise
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Vérification de l'environnement
   import sys
   print(f"Python version: {sys.version}")
   
   # Librairies essentielles
   required_packages = [
       'pandas', 'numpy', 'matplotlib', 'seaborn',
       'tensorflow', 'scikit-learn', 'optuna',
       'streamlit', 'plotly'
   ]
   
   for package in required_packages:
       try:
           __import__(package)
           print(f"✓ {package}")
       except ImportError:
           print(f"✗ {package} - Installation requise")

Dépannage des notebooks
----------------------

Problèmes courants
~~~~~~~~~~~~~~~~~

1. **Erreurs de mémoire**
   
   - Réduire la taille des batches
   - Utiliser des générateurs de données
   - Nettoyer les variables inutiles

2. **Erreurs de chemins**
   
   - Vérifier les chemins relatifs
   - Utiliser os.path.join()
   - Tester l'existence des fichiers

3. **Problèmes de versions**
   
   - Vérifier la compatibilité TensorFlow/Keras
   - Mettre à jour les dépendances
   - Utiliser des environnements virtuels

Solutions type
~~~~~~~~~~~~~

.. code-block:: python

   # Gestion robuste des erreurs
   
   import os
   import warnings
   warnings.filterwarnings('ignore')
   
   # Vérification des chemins
   data_path = '../Data/data.csv'
   if not os.path.exists(data_path):
       print(f"Fichier non trouvé: {data_path}")
       print("Vérifiez le chemin ou exécutez depuis le bon répertoire")
   
   # Gestion mémoire
   import gc
   
   def clean_memory():
       """Nettoyage de la mémoire"""
       gc.collect()
       
   # À utiliser après les sections intensives

Prochaines étapes
-----------------

Les notebooks constituent la base pratique du projet. Pour aller plus loin :

* :doc:`../troubleshooting` - Résolution de problèmes
* :doc:`../api_reference` - Documentation des fonctions
* :doc:`../faq` - Questions fréquentes
