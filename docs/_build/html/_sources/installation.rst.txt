Installation
============

Ce guide vous aidera à installer et configurer l'environnement pour le projet Analysing Green Energy.

Prérequis
---------

Avant de commencer, assurez-vous d'avoir installé :

- **Python 3.8+** (recommandé : Python 3.9 ou 3.10)
- **pip** (gestionnaire de paquets Python)
- **Git** (pour cloner le repository)

Installation rapide
-------------------

1. **Cloner le repository**

.. code-block:: bash

    git clone https://github.com/votre-username/AnalysingEnergy.git
    cd AnalysingEnergy

2. **Créer un environnement virtuel**

.. code-block:: bash

    # Avec venv
    python -m venv env
    
    # Activer l'environnement
    # Windows
    env\Scripts\activate
    # Linux/Mac
    source env/bin/activate

3. **Installer les dépendances**

.. code-block:: bash

    pip install -r requirements.txt

Dépendances principales
-----------------------

Le projet utilise les bibliothèques suivantes :

**Machine Learning & Data Science**
- ``tensorflow`` - Pour les modèles LSTM
- ``scikit-learn`` - Pour le preprocessing et l'évaluation
- ``pandas`` - Manipulation des données
- ``numpy`` - Calculs numériques

**Visualisation**
- ``matplotlib`` - Graphiques de base
- ``seaborn`` - Visualisations statistiques
- ``plotly`` - Graphiques interactifs

**Interface utilisateur**
- ``streamlit`` - Application web interactive

**Optimisation**
- ``optuna`` - Optimisation des hyperparamètres

**Autres**
- ``joblib`` - Sauvegarde/chargement des modèles
- ``statsmodels`` - Analyse des séries temporelles

Installation des dépendances spécifiques
----------------------------------------

Si vous rencontrez des problèmes avec TensorFlow :

.. code-block:: bash

    # Pour CPU seulement
    pip install tensorflow-cpu
    
    # Pour GPU (nécessite CUDA)
    pip install tensorflow-gpu

Pour Jupyter Notebooks :

.. code-block:: bash

    pip install jupyter notebook ipykernel
    python -m ipykernel install --user --name=env

Vérification de l'installation
------------------------------

Pour vérifier que tout est correctement installé :

.. code-block:: python

    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import streamlit as st
    
    print("TensorFlow version:", tf.__version__)
    print("Pandas version:", pd.__version__)
    print("NumPy version:", np.__version__)
    
    # Vérifier la disponibilité du GPU
    print("GPU disponible:", tf.config.list_physical_devices('GPU'))

Structure du projet
-------------------

Après l'installation, votre dossier devrait avoir cette structure :

.. code-block::

    AnalysingEnergy/
    ├── Data/
    │   ├── data.csv
    │   ├── train_data.csv
    │   └── test_data.csv
    ├── Notebooks/
    │   ├── Data preprocessing.ipynb
    │   ├── LSTM Generation.ipynb
    │   ├── LSTM complet.ipynb
    │   ├── Predicting_next_365days.ipynb
    │   ├── models/
    │   └── scalers/
    ├── interface/
    │   └── app.py
    ├── docs/
    │   └── (fichiers de documentation)
    ├── requirements.txt
    └── README.md

Configuration IDE
-----------------

**VS Code**

Extensions recommandées :
- Python
- Jupyter
- Python Docstring Generator

**PyCharm**

Configurer l'interpréteur Python pour pointer vers votre environnement virtuel.

Résolution des problèmes
------------------------

**Erreur de mémoire avec TensorFlow**

.. code-block:: python

    # Ajouter au début de vos scripts
    import tensorflow as tf
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

**Problèmes avec les notebooks**

Si les notebooks ne s'ouvrent pas correctement :

.. code-block:: bash

    jupyter notebook --generate-config
    jupyter notebook

**Erreurs de versions**

Créez un fichier ``requirements.txt`` avec des versions spécifiques :

.. code-block::

    tensorflow==2.13.0
    pandas==2.0.3
    numpy==1.24.3
    scikit-learn==1.3.0
    matplotlib==3.7.2
    streamlit==1.25.0
    optuna==3.2.0

Étapes suivantes
----------------

Une fois l'installation terminée :

1. Consultez le :doc:`quickstart` pour un aperçu rapide
2. Explorez les données avec :doc:`data_description`
3. Lancez votre premier modèle avec :doc:`lstm_models`
