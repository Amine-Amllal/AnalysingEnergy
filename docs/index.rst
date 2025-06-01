.. AnalysingGreenEnergy documentation master file, created by
   sphinx-quickstart on Wed May 21 18:52:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Analysing Green Energy - Documentation
======================================

Bienvenue dans la documentation du projet **Analysing Green Energy**, un système complet d'analyse et de prédiction de la production d'énergie verte utilisant des réseaux de neurones LSTM (Long Short-Term Memory).

Ce projet utilise le dataset **BanE-16** pour prédire la génération maximale d'énergie en fonction de variables météorologiques et de demande énergétique.

🎯 **Objectifs du projet**
--------------------------

- Analyser les données de production d'énergie verte
- Développer des modèles LSTM pour la prédiction de génération d'énergie
- Créer une interface interactive pour visualiser les prédictions
- Optimiser les hyperparamètres des modèles avec Optuna
- Prédire la production d'énergie pour les 365 prochains jours

🚀 **Démarrage rapide**
-----------------------

1. **Installation des dépendances**::

    pip install -r requirements.txt

2. **Exploration des données**:
   Consultez le notebook `Data preprocessing.ipynb`

3. **Entraînement du modèle**:
   Utilisez `LSTM Generation.ipynb`

4. **Interface utilisateur**::

    streamlit run interface/app.py

📚 **Table des matières**
-------------------------

.. toctree::
   :maxdepth: 2
   :caption: Guide d'utilisation

   installation
   quickstart
   data_description
   
.. toctree::
   :maxdepth: 2
   :caption: Analyse des données
   
   data_analysis
   preprocessing
   
.. toctree::
   :maxdepth: 2
   :caption: Modélisation
   
   lstm_models
   hyperparameter_optimization
   model_evaluation

.. toctree::
   :maxdepth: 2
   :caption: Notebooks
   
   notebooks/data_preprocessing
   notebooks/lstm_training
   notebooks/prediction
   
.. toctree::
   :maxdepth: 2
   :caption: Interface utilisateur
   
   interface
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Annexes
   
   troubleshooting
   faq
   changelog

📊 **Aperçu du projet**
-----------------------

Ce projet comprend plusieurs composants clés :

**Données**
- Dataset BanE-16 avec variables météorologiques
- Données de demande et génération d'énergie
- Préprocessing et normalisation des données

**Modèles**
- Réseaux LSTM pour prédiction temporelle
- Optimisation avec Optuna
- Modèles spécialisés par variable

**Interface**
- Application Streamlit interactive
- Visualisations en temps réel
- Comparaison production vs consommation

Indices et tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

