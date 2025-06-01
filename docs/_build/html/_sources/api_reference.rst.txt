Référence API
=============

Cette section fournit la documentation complète de l'API du projet AnalysingEnergy, incluant toutes les classes, fonctions et méthodes principales.

Vue d'ensemble de l'API
-----------------------

L'API du projet est organisée en plusieurs modules principaux :

* **EnergyPredictor** - Classe principale pour les prédictions énergétiques
* **DataPreprocessor** - Utilitaires de prétraitement des données
* **ModelOptimizer** - Optimisation des hyperparamètres
* **Visualizer** - Fonctions de visualisation
* **Utils** - Fonctions utilitaires

Module principal : EnergyPredictor
----------------------------------

.. autoclass:: interface.app.EnergyPredictor
   :members:
   :undoc-members:
   :show-inheritance:

Classe EnergyPredictor
~~~~~~~~~~~~~~~~~~~~~

La classe principale pour l'interface de prédiction énergétique.

.. code-block:: python

   class EnergyPredictor:
       """
       Classe principale pour la prédiction de génération d'énergie verte.
       
       Cette classe encapsule toute la logique de prédiction, du chargement
       des données jusqu'à la génération des prédictions finales.
       
       Attributes:
           data (pd.DataFrame): Données chargées
           models (dict): Dictionnaire des modèles entraînés
           scalers (dict): Scalers pour la normalisation
           predictions (dict): Prédictions stockées
       """

Constructeur
^^^^^^^^^^^

.. py:method:: __init__(self)

   Initialise une nouvelle instance de EnergyPredictor.
   
   :rtype: None
   
   **Exemple:**
   
   .. code-block:: python
   
      predictor = EnergyPredictor()

Méthodes de chargement des données
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: load_data(self, file_path)

   Charge les données depuis un fichier CSV.
   
   :param str file_path: Chemin vers le fichier de données
   :raises FileNotFoundError: Si le fichier n'existe pas
   :raises ValueError: Si le format des données est invalide
   :rtype: bool
   :returns: True si le chargement réussit
   
   **Exemple:**
   
   .. code-block:: python
   
      success = predictor.load_data('Data/data.csv')
      if success:
          print("Données chargées avec succès")

.. py:method:: load_data_from_dataframe(self, dataframe)

   Charge les données depuis un DataFrame pandas.
   
   :param pd.DataFrame dataframe: DataFrame contenant les données
   :raises ValueError: Si les colonnes requises sont manquantes
   :rtype: bool
   
   **Exemple:**
   
   .. code-block:: python
   
      import pandas as pd
      df = pd.read_csv('data.csv')
      predictor.load_data_from_dataframe(df)

Méthodes de prétraitement
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: preprocess_data(self, target_variable='max_generation(mw)')

   Prétraite les données pour l'entraînement des modèles.
   
   :param str target_variable: Nom de la variable cible
   :rtype: tuple
   :returns: (X_processed, y_processed) données prétraitées
   
   **Exemple:**
   
   .. code-block:: python
   
      X, y = predictor.preprocess_data()
      print(f"Shape des données: X={X.shape}, y={y.shape}")

.. py:method:: create_sequences(self, data, sequence_length=60)

   Crée des séquences temporelles pour l'entraînement LSTM.
   
   :param np.array data: Données d'entrée
   :param int sequence_length: Longueur des séquences
   :rtype: tuple
   :returns: (X_sequences, y_sequences)
   
   **Exemple:**
   
   .. code-block:: python
   
      X_seq, y_seq = predictor.create_sequences(data, sequence_length=60)

Méthodes d'entraînement
^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: train_models(self, epochs=100, batch_size=32)

   Entraîne les modèles LSTM sur les données préparées.
   
   :param int epochs: Nombre d'époques d'entraînement
   :param int batch_size: Taille des batches
   :rtype: dict
   :returns: Historique d'entraînement
   
   **Exemple:**
   
   .. code-block:: python
   
      history = predictor.train_models(epochs=150, batch_size=16)
      print(f"Loss finale: {history['loss'][-1]:.4f}")

.. py:method:: load_trained_models(self, models_dir='Notebooks/models/')

   Charge des modèles pré-entraînés depuis le disque.
   
   :param str models_dir: Répertoire contenant les modèles
   :rtype: bool
   :returns: True si le chargement réussit
   
   **Exemple:**
   
   .. code-block:: python
   
      if predictor.load_trained_models():
          print("Modèles chargés avec succès")

Méthodes de prédiction
^^^^^^^^^^^^^^^^^^^^^

.. py:method:: predict_single_day(self, min_temp, mean_temp, max_temp, wind_speed, precipitation, pressure, humidity)

   Effectue une prédiction pour une journée avec des paramètres météorologiques.
   
   :param float min_temp: Température minimale (°C)
   :param float mean_temp: Température moyenne (°C)
   :param float max_temp: Température maximale (°C)
   :param float wind_speed: Vitesse du vent (m/s)
   :param float precipitation: Précipitations (mm)
   :param float pressure: Pression atmosphérique (Pa)
   :param float humidity: Humidité relative (%)
   :rtype: float
   :returns: Prédiction de génération (MW)
   
   **Exemple:**
   
   .. code-block:: python
   
      prediction = predictor.predict_single_day(
          min_temp=15.2, mean_temp=22.1, max_temp=28.9,
          wind_speed=12.5, precipitation=0.0,
          pressure=1013.25, humidity=65.0
      )
      print(f"Génération prédite: {prediction:.2f} MW")

.. py:method:: predict_future(self, days=30)

   Génère des prédictions pour les prochains jours.
   
   :param int days: Nombre de jours à prédire
   :rtype: np.array
   :returns: Array des prédictions
   
   **Exemple:**
   
   .. code-block:: python
   
      predictions = predictor.predict_future(days=7)
      print(f"Prédictions pour 7 jours: {predictions}")

Méthodes d'évaluation
^^^^^^^^^^^^^^^^^^^^

.. py:method:: evaluate_models(self, test_data=None)

   Évalue les performances des modèles entraînés.
   
   :param pd.DataFrame test_data: Données de test (optionnel)
   :rtype: dict
   :returns: Métriques d'évaluation
   
   **Exemple:**
   
   .. code-block:: python
   
      metrics = predictor.evaluate_models()
      print(f"RMSE: {metrics['RMSE']:.2f}")
      print(f"MAE: {metrics['MAE']:.2f}")

.. py:method:: calculate_metrics(self, y_true, y_pred)

   Calcule les métriques de performance.
   
   :param np.array y_true: Valeurs réelles
   :param np.array y_pred: Valeurs prédites
   :rtype: dict
   :returns: Dictionnaire des métriques
   
   **Exemple:**
   
   .. code-block:: python
   
      metrics = predictor.calculate_metrics(y_true, y_pred)
      print(f"R²: {metrics['R2']:.4f}")

Méthodes de visualisation
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: plot_predictions(self, save_path=None)

   Génère des graphiques de prédictions vs valeurs réelles.
   
   :param str save_path: Chemin de sauvegarde (optionnel)
   :rtype: None
   
   **Exemple:**
   
   .. code-block:: python
   
      predictor.plot_predictions(save_path='predictions.png')

.. py:method:: plot_training_history(self, history, save_path=None)

   Visualise l'historique d'entraînement.
   
   :param dict history: Historique d'entraînement
   :param str save_path: Chemin de sauvegarde (optionnel)
   :rtype: None

Classes utilitaires
-------------------

DataPreprocessor
~~~~~~~~~~~~~~~

.. py:class:: DataPreprocessor

   Classe pour le prétraitement des données.

.. py:method:: __init__(self, sequence_length=60, target_variable='max_generation(mw)')

   :param int sequence_length: Longueur des séquences
   :param str target_variable: Variable cible

.. py:method:: fit_transform(self, data)

   Pipeline complet de prétraitement.
   
   :param pd.DataFrame data: Données brutes
   :rtype: tuple
   :returns: (X, y) données prétraitées

.. py:method:: clean_data(self, data)

   Nettoyage des données.
   
   :param pd.DataFrame data: Données à nettoyer
   :rtype: pd.DataFrame

.. py:method:: create_features(self, data)

   Création de features temporelles et dérivées.
   
   :param pd.DataFrame data: Données d'entrée
   :rtype: pd.DataFrame

.. py:method:: scale_features(self, data)

   Normalisation des features.
   
   :param pd.DataFrame data: Données à normaliser
   :rtype: pd.DataFrame

ModelOptimizer
~~~~~~~~~~~~~

.. py:class:: ModelOptimizer

   Classe pour l'optimisation des hyperparamètres avec Optuna.

.. py:method:: __init__(self, X_train, y_train, X_val, y_val)

   :param np.array X_train: Données d'entraînement
   :param np.array y_train: Labels d'entraînement
   :param np.array X_val: Données de validation
   :param np.array y_val: Labels de validation

.. py:method:: objective(self, trial)

   Fonction objectif pour l'optimisation.
   
   :param optuna.Trial trial: Trial Optuna
   :rtype: float
   :returns: Métrique à optimiser

.. py:method:: optimize(self, n_trials=100)

   Lance l'optimisation des hyperparamètres.
   
   :param int n_trials: Nombre d'essais
   :rtype: dict
   :returns: Meilleurs hyperparamètres

Visualizer
~~~~~~~~~

.. py:class:: Visualizer

   Classe pour les visualisations avancées.

.. py:method:: plot_time_series(self, data, columns, title="Série temporelle")

   Graphique de séries temporelles.
   
   :param pd.DataFrame data: Données
   :param list columns: Colonnes à afficher
   :param str title: Titre du graphique

.. py:method:: plot_correlation_matrix(self, data, save_path=None)

   Matrice de corrélation.
   
   :param pd.DataFrame data: Données
   :param str save_path: Chemin de sauvegarde

.. py:method:: plot_residual_analysis(self, y_true, y_pred)

   Analyse des résidus.
   
   :param np.array y_true: Valeurs réelles
   :param np.array y_pred: Valeurs prédites

Fonctions utilitaires
--------------------

Gestion des données
~~~~~~~~~~~~~~~~~~

.. py:function:: load_dataset(file_path, validate=True)

   Charge un dataset avec validation.
   
   :param str file_path: Chemin du fichier
   :param bool validate: Activer la validation
   :rtype: pd.DataFrame
   :raises FileNotFoundError: Si le fichier n'existe pas
   
   **Exemple:**
   
   .. code-block:: python
   
      data = load_dataset('Data/data.csv')

.. py:function:: validate_data_format(data)

   Valide le format des données d'entrée.
   
   :param pd.DataFrame data: Données à valider
   :rtype: bool
   :raises ValueError: Si le format est incorrect

.. py:function:: split_train_test(data, test_size=0.2, time_based=True)

   Division des données en train/test.
   
   :param pd.DataFrame data: Données
   :param float test_size: Proportion de test
   :param bool time_based: Division temporelle
   :rtype: tuple

Utilitaires de modélisation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: create_lstm_model(input_shape, **kwargs)

   Crée un modèle LSTM avec les paramètres spécifiés.
   
   :param tuple input_shape: Forme d'entrée
   :param kwargs: Hyperparamètres du modèle
   :rtype: tf.keras.Model
   
   **Paramètres disponibles:**
   
   * units_1 (int): Unités première couche LSTM
   * units_2 (int): Unités deuxième couche LSTM  
   * dropout_rate (float): Taux de dropout
   * activation (str): Fonction d'activation
   * learning_rate (float): Taux d'apprentissage

.. py:function:: save_model_with_metadata(model, filepath, metadata=None)

   Sauvegarde un modèle avec ses métadonnées.
   
   :param tf.keras.Model model: Modèle à sauvegarder
   :param str filepath: Chemin de sauvegarde
   :param dict metadata: Métadonnées additionnelles

.. py:function:: load_model_with_metadata(filepath)

   Charge un modèle avec ses métadonnées.
   
   :param str filepath: Chemin du modèle
   :rtype: tuple
   :returns: (model, metadata)

Utilitaires de visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: create_plotly_chart(data, chart_type='line', **kwargs)

   Crée un graphique Plotly interactif.
   
   :param pd.DataFrame data: Données
   :param str chart_type: Type de graphique
   :rtype: plotly.graph_objects.Figure

.. py:function:: save_figure(fig, filepath, format='png', dpi=300)

   Sauvegarde une figure avec les paramètres spécifiés.
   
   :param matplotlib.figure.Figure fig: Figure à sauvegarder
   :param str filepath: Chemin de sauvegarde
   :param str format: Format de fichier
   :param int dpi: Résolution

Gestion des erreurs
------------------

Exceptions personnalisées
~~~~~~~~~~~~~~~~~~~~~~~~

.. py:exception:: DataValidationError

   Erreur de validation des données.
   
   Levée quand les données d'entrée ne respectent pas le format attendu.

.. py:exception:: ModelNotTrainedError

   Erreur de modèle non entraîné.
   
   Levée quand on tente d'utiliser un modèle non entraîné.

.. py:exception:: PredictionError

   Erreur de prédiction.
   
   Levée lors d'une erreur pendant la génération de prédictions.

Exemples d'utilisation
----------------------

Utilisation basique
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Import de la classe principale
   from interface.app import EnergyPredictor
   
   # Initialisation
   predictor = EnergyPredictor()
   
   # Chargement des données
   predictor.load_data('Data/data.csv')
   
   # Chargement des modèles pré-entraînés
   predictor.load_trained_models()
   
   # Prédiction simple
   prediction = predictor.predict_single_day(
       min_temp=18.0, mean_temp=25.0, max_temp=32.0,
       wind_speed=15.0, precipitation=0.0,
       pressure=1013.0, humidity=60.0
   )
   
   print(f"Génération prédite: {prediction:.2f} MW")

Entraînement complet
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Pipeline complet d'entraînement
   predictor = EnergyPredictor()
   predictor.load_data('Data/data.csv')
   
   # Prétraitement
   X, y = predictor.preprocess_data()
   
   # Entraînement
   history = predictor.train_models(epochs=200, batch_size=16)
   
   # Évaluation
   metrics = predictor.evaluate_models()
   print(f"Performance: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
   
   # Visualisation
   predictor.plot_predictions()
   predictor.plot_training_history(history)

Optimisation des hyperparamètres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from your_module import ModelOptimizer
   
   # Préparation des données
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
   
   # Optimisation
   optimizer = ModelOptimizer(X_train, y_train, X_val, y_val)
   best_params = optimizer.optimize(n_trials=50)
   
   print(f"Meilleurs paramètres: {best_params}")

Configuration et constantes
---------------------------

Constantes du modèle
~~~~~~~~~~~~~~~~~~~

.. py:data:: DEFAULT_SEQUENCE_LENGTH
   :type: int
   :value: 60
   
   Longueur par défaut des séquences LSTM.

.. py:data:: WEATHER_VARIABLES
   :type: list
   
   Liste des variables météorologiques d'entrée.
   
   .. code-block:: python
   
      WEATHER_VARIABLES = [
          'min_temperature', 'mean_temperature', 'max_temperature',
          'min_windspeed', 'mean_windspeed', 'max_windspeed',
          'total_precipitation', 'surface_pressure', 'mean_relative_humidity'
      ]

.. py:data:: TARGET_VARIABLE
   :type: str
   :value: 'max_generation(mw)'
   
   Variable cible par défaut.

Paramètres du modèle optimisé
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:data:: OPTIMAL_HYPERPARAMETERS
   :type: dict
   
   Hyperparamètres optimaux trouvés par Optuna.
   
   .. code-block:: python
   
      OPTIMAL_HYPERPARAMETERS = {
          'units_1': 74,
          'units_2': 69,
          'dropout_rate': 0.1938,
          'activation': 'relu',
          'learning_rate': 0.001,
          'batch_size': 32
      }

Notes de version
---------------

Version 1.0.0
~~~~~~~~~~~~~

- Implémentation initiale de l'API
- Support des modèles LSTM basiques
- Interface Streamlit fonctionnelle

Version 1.1.0
~~~~~~~~~~~~~

- Ajout de l'optimisation Optuna
- Amélioration des visualisations
- Support des prédictions long terme

Version actuelle
~~~~~~~~~~~~~~~

- API complète et documentée
- Modèles optimisés (RMSE: 291.19 MW)
- Interface utilisateur avancée
- Documentation complète
