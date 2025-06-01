Référence des modules API
========================

Cette section présente la documentation complète de l'API des modules Python du projet AnalysingEnergy.

Vue d'ensemble
--------------

Le projet est organisé en plusieurs modules Python qui fournissent des fonctionnalités pour :

- Préprocessing des données énergétiques
- Entraînement et évaluation des modèles LSTM
- Interface utilisateur avec Streamlit
- Utilitaires et fonctions helpers

Architecture des modules
------------------------

Structure du code
~~~~~~~~~~~~~~~~

.. code-block:: text

    AnalysingEnergy/
    ├── interface/
    │   ├── app.py              # Application Streamlit principale
    │   └── utils.py            # Utilitaires pour l'interface
    ├── src/
    │   ├── data/
    │   │   ├── loader.py       # Chargement des données
    │   │   ├── preprocessor.py # Préprocessing
    │   │   └── validator.py    # Validation des données
    │   ├── models/
    │   │   ├── lstm.py         # Modèles LSTM
    │   │   ├── trainer.py      # Entraînement
    │   │   └── evaluator.py    # Évaluation
    │   └── utils/
    │       ├── config.py       # Configuration
    │       ├── metrics.py      # Métriques
    │       └── visualization.py # Visualisations
    └── Notebooks/              # Notebooks Jupyter

Module interface
---------------

interface.app
~~~~~~~~~~~~~

Module principal de l'application Streamlit.

.. automodule:: interface.app
   :members:
   :undoc-members:
   :show-inheritance:

**Classes principales :**

.. autoclass:: interface.app.EnergyAnalyzer
   :members:
   :undoc-members:

**Fonctions utilitaires :**

.. autofunction:: interface.app.load_data

.. autofunction:: interface.app.create_visualizations

.. autofunction:: interface.app.export_results

**Configuration de l'application :**

.. code-block:: python

    import streamlit as st
    from interface.app import EnergyAnalyzer
    
    # Configuration de base
    st.set_page_config(
        page_title="🔋 Analyse Production vs Consommation", 
        layout="centered"
    )
    
    # Initialisation de l'analyseur
    analyzer = EnergyAnalyzer()
    analyzer.run()

interface.utils
~~~~~~~~~~~~~~

Utilitaires pour l'interface utilisateur.

.. automodule:: interface.utils
   :members:
   :undoc-members:
   :show-inheritance:

**Fonctions de visualisation :**

.. autofunction:: interface.utils.plot_energy_comparison

.. autofunction:: interface.utils.create_dashboard_metrics

.. autofunction:: interface.utils.generate_report

Module data
----------

src.data.loader
~~~~~~~~~~~~~~

Chargement et gestion des données énergétiques.

.. automodule:: src.data.loader
   :members:
   :undoc-members:
   :show-inheritance:

**Classe DataLoader :**

.. autoclass:: src.data.loader.DataLoader
   :members:
   :undoc-members:

   .. automethod:: load_csv_data
   .. automethod:: load_time_series
   .. automethod:: validate_data_format

**Exemple d'utilisation :**

.. code-block:: python

    from src.data.loader import DataLoader
    
    # Initialisation
    loader = DataLoader('Data/data.csv')
    
    # Chargement des données
    data = loader.load_csv_data()
    
    # Validation
    is_valid = loader.validate_data_format(data)

src.data.preprocessor
~~~~~~~~~~~~~~~~~~~~

Préprocessing des données pour les modèles LSTM.

.. automodule:: src.data.preprocessor
   :members:
   :undoc-members:
   :show-inheritance:

**Classe EnergyDataPreprocessor :**

.. autoclass:: src.data.preprocessor.EnergyDataPreprocessor
   :members:
   :undoc-members:

   .. automethod:: clean_data
   .. automethod:: handle_missing_values
   .. automethod:: detect_outliers
   .. automethod:: normalize_features
   .. automethod:: create_sequences

**Exemple de preprocessing :**

.. code-block:: python

    from src.data.preprocessor import EnergyDataPreprocessor
    
    # Initialisation
    preprocessor = EnergyDataPreprocessor()
    
    # Nettoyage
    clean_data = preprocessor.clean_data(raw_data)
    
    # Normalisation
    normalized_data = preprocessor.normalize_features(clean_data)
    
    # Création de séquences
    X_sequences, y_sequences = preprocessor.create_sequences(
        normalized_data, time_steps=60
    )

src.data.validator
~~~~~~~~~~~~~~~~~

Validation de la qualité des données.

.. automodule:: src.data.validator
   :members:
   :undoc-members:
   :show-inheritance:

**Classe DataValidator :**

.. autoclass:: src.data.validator.DataValidator
   :members:
   :undoc-members:

   .. automethod:: check_data_completeness
   .. automethod:: validate_ranges
   .. automethod:: check_temporal_consistency
   .. automethod:: generate_quality_report

Module models
------------

src.models.lstm
~~~~~~~~~~~~~~

Modèles LSTM pour la prédiction énergétique.

.. automodule:: src.models.lstm
   :members:
   :undoc-members:
   :show-inheritance:

**Classe LSTMEnergyModel :**

.. autoclass:: src.models.lstm.LSTMEnergyModel
   :members:
   :undoc-members:

   .. automethod:: build_model
   .. automethod:: compile_model
   .. automethod:: predict
   .. automethod:: predict_sequence
   .. automethod:: save_model
   .. automethod:: load_model

**Architecture du modèle :**

.. code-block:: python

    from src.models.lstm import LSTMEnergyModel
    
    # Création du modèle
    model = LSTMEnergyModel(
        input_shape=(60, 9),
        lstm_units=[64, 32],
        dropout_rate=0.2
    )
    
    # Construction
    model.build_model()
    
    # Compilation
    model.compile_model(learning_rate=0.001)

src.models.trainer
~~~~~~~~~~~~~~~~~

Entraînement des modèles LSTM.

.. automodule:: src.models.trainer
   :members:
   :undoc-members:
   :show-inheritance:

**Classe ModelTrainer :**

.. autoclass:: src.models.trainer.ModelTrainer
   :members:
   :undoc-members:

   .. automethod:: setup_callbacks
   .. automethod:: train_model
   .. automethod:: cross_validate
   .. automethod:: hyperparameter_tuning
   .. automethod:: save_training_history

**Exemple d'entraînement :**

.. code-block:: python

    from src.models.trainer import ModelTrainer
    from src.models.lstm import LSTMEnergyModel
    
    # Préparation
    model = LSTMEnergyModel(input_shape=(60, 9))
    trainer = ModelTrainer(model)
    
    # Entraînement
    history = trainer.train_model(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32
    )

src.models.evaluator
~~~~~~~~~~~~~~~~~~~

Évaluation des performances des modèles.

.. automodule:: src.models.evaluator
   :members:
   :undoc-members:
   :show-inheritance:

**Classe ModelEvaluator :**

.. autoclass:: src.models.evaluator.ModelEvaluator
   :members:
   :undoc-members:

   .. automethod:: evaluate_model
   .. automethod:: calculate_metrics
   .. automethod:: plot_predictions
   .. automethod:: analyze_residuals
   .. automethod:: generate_evaluation_report

Module utils
-----------

src.utils.config
~~~~~~~~~~~~~~~~

Configuration globale du projet.

.. automodule:: src.utils.config
   :members:
   :undoc-members:
   :show-inheritance:

**Configuration par défaut :**

.. code-block:: python

    from src.utils.config import Config
    
    # Accès aux paramètres
    config = Config()
    
    # Paramètres de modèle
    model_params = config.MODEL_PARAMETERS
    
    # Chemins de fichiers
    data_path = config.DATA_PATH
    model_path = config.MODEL_PATH

src.utils.metrics
~~~~~~~~~~~~~~~~~

Métriques d'évaluation personnalisées.

.. automodule:: src.utils.metrics
   :members:
   :undoc-members:
   :show-inheritance:

**Fonctions de métriques :**

.. autofunction:: src.utils.metrics.calculate_rmse

.. autofunction:: src.utils.metrics.calculate_mape

.. autofunction:: src.utils.metrics.calculate_directional_accuracy

.. autofunction:: src.utils.metrics.energy_specific_metrics

**Exemple d'utilisation :**

.. code-block:: python

    from src.utils.metrics import calculate_rmse, calculate_mape
    
    # Calcul des métriques
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAPE: {mape:.2f}%")

src.utils.visualization
~~~~~~~~~~~~~~~~~~~~~~

Utilitaires de visualisation.

.. automodule:: src.utils.visualization
   :members:
   :undoc-members:
   :show-inheritance:

**Fonctions de visualisation :**

.. autofunction:: src.utils.visualization.plot_time_series

.. autofunction:: src.utils.visualization.plot_prediction_comparison

.. autofunction:: src.utils.visualization.plot_correlation_matrix

.. autofunction:: src.utils.visualization.create_dashboard_plot

API Reference complète
----------------------

Classes principales
~~~~~~~~~~~~~~~~~~

**EnergyAnalyzer**

Classe principale pour l'analyse énergétique.

.. code-block:: python

    class EnergyAnalyzer:
        """
        Analyseur principal pour les données énergétiques.
        
        Attributes:
            data_loader (DataLoader): Chargeur de données
            preprocessor (EnergyDataPreprocessor): Préprocesseur
            model (LSTMEnergyModel): Modèle LSTM
            evaluator (ModelEvaluator): Évaluateur
        """
        
        def __init__(self, config_path=None):
            """Initialise l'analyseur avec configuration optionnelle."""
            pass
        
        def load_data(self, file_path):
            """Charge les données depuis un fichier."""
            pass
        
        def preprocess_data(self):
            """Préprocesse les données chargées."""
            pass
        
        def train_model(self, **kwargs):
            """Entraîne le modèle LSTM."""
            pass
        
        def predict(self, input_data):
            """Génère des prédictions."""
            pass
        
        def evaluate(self, test_data):
            """Évalue les performances du modèle."""
            pass

**LSTMEnergyModel**

Modèle LSTM spécialisé pour l'énergie.

.. code-block:: python

    class LSTMEnergyModel:
        """
        Modèle LSTM pour prédiction énergétique.
        
        Attributes:
            input_shape (tuple): Forme des données d'entrée
            lstm_units (list): Unités LSTM par couche
            dropout_rate (float): Taux de dropout
            model (tf.keras.Model): Modèle Keras
        """
        
        def build_model(self):
            """Construit l'architecture du modèle."""
            pass
        
        def compile_model(self, optimizer='adam', loss='mse'):
            """Compile le modèle avec optimiseur et loss."""
            pass
        
        def fit(self, X, y, **kwargs):
            """Entraîne le modèle."""
            pass
        
        def predict(self, X):
            """Génère des prédictions."""
            pass

Fonctions utilitaires
~~~~~~~~~~~~~~~~~~~

**Chargement de données :**

.. autofunction:: src.data.loader.load_energy_data

.. autofunction:: src.data.loader.load_weather_data

**Préprocessing :**

.. autofunction:: src.data.preprocessor.normalize_energy_data

.. autofunction:: src.data.preprocessor.create_lstm_sequences

**Métriques :**

.. autofunction:: src.utils.metrics.energy_forecast_accuracy

.. autofunction:: src.utils.metrics.peak_detection_accuracy

Exemples d'utilisation complète
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pipeline complet :**

.. code-block:: python

    from src.data.loader import DataLoader
    from src.data.preprocessor import EnergyDataPreprocessor
    from src.models.lstm import LSTMEnergyModel
    from src.models.trainer import ModelTrainer
    from src.models.evaluator import ModelEvaluator
    
    # 1. Chargement des données
    loader = DataLoader('Data/data.csv')
    data = loader.load_csv_data()
    
    # 2. Préprocessing
    preprocessor = EnergyDataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.prepare_data(data)
    
    # 3. Création du modèle
    model = LSTMEnergyModel(input_shape=X_train.shape[1:])
    model.build_model()
    
    # 4. Entraînement
    trainer = ModelTrainer(model)
    history = trainer.train_model(X_train, y_train)
    
    # 5. Évaluation
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate_model(X_test, y_test)
    
    print(f"Performance du modèle: {metrics}")

**Interface Streamlit :**

.. code-block:: python

    import streamlit as st
    from interface.app import EnergyAnalyzer
    
    # Configuration Streamlit
    st.set_page_config(page_title="Analyse Énergétique")
    
    # Initialisation
    analyzer = EnergyAnalyzer()
    
    # Interface utilisateur
    uploaded_file = st.file_uploader("Chargez vos données")
    if uploaded_file:
        analyzer.load_data(uploaded_file)
        analyzer.preprocess_data()
        
        if st.button("Entraîner le modèle"):
            analyzer.train_model()
            st.success("Modèle entraîné avec succès!")
        
        if st.button("Générer des prédictions"):
            predictions = analyzer.predict()
            st.line_chart(predictions)

Configuration et déploiement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Variables d'environnement :**

.. code-block:: bash

    # Configuration pour production
    export ENERGY_DATA_PATH="/path/to/data"
    export MODEL_SAVE_PATH="/path/to/models"
    export STREAMLIT_PORT=8501
    export TF_CPP_MIN_LOG_LEVEL=2

**Configuration Docker :**

.. code-block:: dockerfile

    FROM python:3.9-slim
    
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    COPY . .
    
    ENV STREAMLIT_SERVER_PORT=8501
    EXPOSE 8501
    
    CMD ["streamlit", "run", "interface/app.py"]

Tests et validation
~~~~~~~~~~~~~~~~~~

**Tests unitaires :**

.. code-block:: python

    import unittest
    from src.data.preprocessor import EnergyDataPreprocessor
    
    class TestEnergyDataPreprocessor(unittest.TestCase):
        
        def setUp(self):
            self.preprocessor = EnergyDataPreprocessor()
        
        def test_normalize_features(self):
            # Test de normalisation
            pass
        
        def test_create_sequences(self):
            # Test de création de séquences
            pass

**Tests d'intégration :**

.. code-block:: python

    def test_full_pipeline():
        """Test du pipeline complet."""
        # Chargement -> Preprocessing -> Entraînement -> Évaluation
        pass

Notes de développement
~~~~~~~~~~~~~~~~~~~~~

**Conventions de code :**

- Utilisation de type hints Python
- Documentation avec docstrings Google style
- Tests avec pytest
- Formatage avec black
- Linting avec flake8

**Structure des docstrings :**

.. code-block:: python

    def example_function(param1: str, param2: int) -> float:
        """
        Fonction d'exemple avec documentation complète.
        
        Args:
            param1: Description du premier paramètre
            param2: Description du second paramètre
        
        Returns:
            Description de la valeur de retour
        
        Raises:
            ValueError: Description de l'exception
        
        Example:
            >>> result = example_function("test", 42)
            >>> print(result)
            3.14
        """
        pass

Cette documentation API est générée automatiquement à partir du code source et maintenue à jour avec les dernières versions des modules.
