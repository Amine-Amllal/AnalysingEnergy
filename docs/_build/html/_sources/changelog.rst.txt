Changelog
=========

Toutes les modifications notables du projet AnalysingEnergy sont documentées dans ce fichier.

Le format est basé sur `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ et ce projet adhère au `Versioning Sémantique <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Ajouté
~~~~~~
- Support pour données horaires (en développement)
- Modèles ensemble pour améliorer la robustesse
- API REST pour intégration externe
- Prédictions avec intervalles de confiance

Amélioré
~~~~~~~~
- Performance des prédictions long terme
- Interface utilisateur avec plus d'options de visualisation
- Documentation avec exemples interactifs

[1.2.0] - 2025-06-01
---------------------

Ajouté
~~~~~~
- Documentation complète ReadTheDocs
- Guide de démarrage rapide
- Section de dépannage détaillée
- FAQ complète
- Documentation API exhaustive
- Notebooks détaillés avec explications

Amélioré
~~~~~~~~
- Structure de la documentation
- Configuration Sphinx avec thème ReadTheDocs
- Support nbsphinx pour les notebooks
- Intégration myst_parser pour Markdown

Corrigé
~~~~~~~
- Liens de navigation dans la documentation
- Formatage des exemples de code
- Références croisées entre sections

[1.1.0] - 2025-05-15
---------------------

Ajouté
~~~~~~
- Optimisation des hyperparamètres avec Optuna
- Modèle final optimisé (RMSE: 291.19 MW)
- Interface Streamlit complète
- Visualisations interactives avec Plotly
- Prédictions à long terme (365 jours)
- Sauvegarde automatique des modèles et scalers

Amélioré
~~~~~~~~
- Performance du modèle LSTM (amélioration de 15% du RMSE)
- Pipeline de prétraitement des données
- Gestion des valeurs manquantes
- Interface utilisateur plus intuitive

Corrigé
~~~~~~~
- Problèmes de mémoire lors de l'entraînement
- Erreurs de normalisation des données
- Bugs dans l'interface Streamlit

[1.0.0] - 2025-04-01
---------------------

Ajouté
~~~~~~
- Modèle LSTM de base pour prédiction énergétique
- Pipeline de prétraitement des données BanE-16
- Interface Streamlit basique
- Notebooks Jupyter pour l'analyse
- Métriques d'évaluation (RMSE, MAE, R²)
- Visualisations avec Matplotlib

Fonctionnalités principales
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Prédiction de génération d'énergie verte
- Analyse des données météorologiques
- Entraînement de modèles LSTM
- Interface web pour les prédictions
- Sauvegarde des modèles entraînés

Architecture
~~~~~~~~~~~~
- Classe EnergyPredictor principale
- Modules de prétraitement des données
- Utilitaires de visualisation
- Structure modulaire du code

[0.3.0] - 2025-03-15
---------------------

Ajouté
~~~~~~
- Modèles LSTM spécialisés par variable météorologique
- Validation croisée temporelle
- Analyse des corrélations variables-génération
- Export des résultats en CSV

Amélioré
~~~~~~~~
- Précision des prédictions (RMSE < 350 MW)
- Stabilité de l'entraînement
- Gestion des outliers

[0.2.0] - 2025-02-20
---------------------

Ajouté
~~~~~~
- Prétraitement avancé des données
- Feature engineering temporel
- Normalisation des variables
- Création de séquences pour LSTM

Amélioré
~~~~~~~~
- Qualité des données d'entrée
- Pipeline de transformation
- Gestion des valeurs aberrantes

[0.1.0] - 2025-01-10
---------------------

Ajouté
~~~~~~
- Structure initiale du projet
- Chargement du dataset BanE-16
- Analyse exploratoire des données
- Premier modèle LSTM basique
- Notebooks d'exploration

Fonctionnalités initiales
~~~~~~~~~~~~~~~~~~~~~~~~
- Lecture des données CSV
- Statistiques descriptives
- Graphiques d'exploration
- Modèle proof-of-concept

Détails des versions
-------------------

Version 1.2.0 - Documentation complète
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cette version majeure introduit une documentation complète et professionnelle :

**Documentation ReadTheDocs :**

- Configuration Sphinx optimisée
- Thème ReadTheDocs moderne
- Support multi-format (RST, Markdown, Notebooks)
- Navigation intuitive et recherche intégrée

**Contenu documentaire :**

- Guide d'installation détaillé
- Démarrage rapide en 4 étapes
- Description complète du dataset BanE-16
- Documentation des modèles LSTM
- Guide d'optimisation Optuna
- Interface Streamlit expliquée
- Analyse des données approfondie
- Prétraitement step-by-step
- Évaluation des modèles
- Documentation des notebooks
- Référence API complète
- Dépannage et FAQ

**Améliorations techniques :**

- Extensions Sphinx : autodoc, napoleon, nbsphinx, myst_parser
- Génération automatique de la documentation API
- Intégration des notebooks Jupyter
- Support du markup Markdown
- Configuration pour ReadTheDocs

Version 1.1.0 - Optimisation et interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version majeure avec optimisation avancée des performances :

**Optimisation Optuna :**

- Optimisation multi-objectif (RMSE + temps d'entraînement)
- 100+ trials d'optimisation automatique
- Hyperparamètres optimaux découverts :
  
  - units_1: 74
  - units_2: 69  
  - dropout_rate: 0.1938
  - activation: 'relu'

**Performance du modèle :**

- RMSE final : 291.19 MW (amélioration de 15%)
- R² : 0.847
- MAE : ~185 MW
- Skill Score : 0.73 vs persistence

**Interface Streamlit :**

- Interface utilisateur complète et intuitive
- Prédictions en temps réel
- Visualisations interactives Plotly
- Upload de fichiers personnalisés
- Export des résultats

**Fonctionnalités avancées :**

- Prédictions long terme (365 jours)
- Analyse des résidus
- Comparaison avec modèles baseline
- Monitoring des performances

Version 1.0.0 - Release stable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Première version stable du système de prédiction :

**Architecture LSTM :**

- Modèle à 2 couches LSTM
- Séquences d'entrée de 60 jours
- Dropout pour régularisation
- Optimiseur Adam

**Pipeline de données :**

- Prétraitement automatisé
- Normalisation MinMaxScaler/StandardScaler
- Création de features temporelles
- Division chronologique train/test

**Métriques d'évaluation :**

- RMSE, MAE, R², MAPE
- Validation temporelle
- Analyse des corrélations
- Tests de robustesse

**Interface et visualisations :**

- Application Streamlit fonctionnelle
- Graphiques Matplotlib
- Export des prédictions
- Interface de configuration

Notes de migration
------------------

Migration vers 1.2.0
~~~~~~~~~~~~~~~~~~~~

Aucune modification de code nécessaire. La nouvelle version ajoute uniquement la documentation.

**Actions recommandées :**

1. Mettre à jour les dépendances pour la documentation :

   .. code-block:: bash
   
      pip install sphinx sphinx-rtd-theme myst-parser nbsphinx

2. Consulter la nouvelle documentation pour les meilleures pratiques

Migration vers 1.1.0
~~~~~~~~~~~~~~~~~~~~

**Changements dans l'API :**

- Nouveaux paramètres optionnels dans `EnergyPredictor`
- Méthodes d'optimisation ajoutées
- Interface Streamlit restructurée

**Actions requises :**

1. Mettre à jour les imports :

   .. code-block:: python
   
      # Nouveau
      from interface.app import EnergyPredictor
      
      # Si vous utilisez l'optimisation
      from your_module import ModelOptimizer

2. Utiliser les nouveaux modèles optimisés :

   .. code-block:: python
   
      # Charger le modèle optimisé
      predictor.load_trained_models('Notebooks/models/')

Migration vers 1.0.0
~~~~~~~~~~~~~~~~~~~~

**Restructuration majeure :**

- Nouvelle architecture modulaire
- API EnergyPredictor standardisée
- Nouveaux formats de sauvegarde

**Migration depuis 0.x :**

1. Réentraîner les modèles avec la nouvelle architecture
2. Adapter le code pour utiliser la classe EnergyPredictor
3. Mettre à jour les chemins de fichiers

Dépendances par version
----------------------

Version 1.2.0
~~~~~~~~~~~~~

.. code-block:: text

   # Requirements pour la documentation
   sphinx>=7.1.0
   sphinx-rtd-theme>=1.3.0
   myst-parser>=2.0.0
   nbsphinx>=0.9.0
   
   # Requirements existants
   pandas>=1.5.0
   numpy>=1.24.0
   tensorflow>=2.13.0
   streamlit>=1.28.0
   plotly>=5.17.0
   optuna>=3.4.0

Version 1.1.0
~~~~~~~~~~~~~

.. code-block:: text

   pandas>=1.5.0
   numpy>=1.24.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   scikit-learn>=1.3.0
   tensorflow>=2.13.0
   streamlit>=1.28.0
   plotly>=5.17.0
   optuna>=3.4.0

Version 1.0.0
~~~~~~~~~~~~~

.. code-block:: text

   pandas>=1.5.0
   numpy>=1.24.0
   matplotlib>=3.7.0
   scikit-learn>=1.3.0
   tensorflow>=2.13.0
   streamlit>=1.25.0

Problèmes connus
---------------

Version 1.2.0
~~~~~~~~~~~~~

- Aucun problème connu majeur
- Documentation en cours de finalisation

Version 1.1.0
~~~~~~~~~~~~~

- Performance sur CPU peut être lente pour l'optimisation (utiliser GPU recommandé)
- Interface Streamlit peut nécessiter un redémarrage après changement de modèle

Version 1.0.0
~~~~~~~~~~~~~

- Modèle peut avoir des difficultés avec des conditions météo extrêmes
- Interface basique avec options limitées

Contributeurs
------------

**Version 1.2.0 :**
- Documentation complète et professionnelle

**Version 1.1.0 :**
- Optimisation Optuna
- Interface Streamlit avancée  
- Améliorations de performance

**Version 1.0.0 :**
- Architecture LSTM de base
- Pipeline de données
- Interface initiale

Remerciements
------------

- Dataset BanE-16 pour les données d'entraînement
- Communauté TensorFlow pour les outils de deep learning
- Optuna pour l'optimisation des hyperparamètres
- Streamlit pour l'interface utilisateur
- Sphinx et ReadTheDocs pour la documentation

Liens utiles
-----------

- `Repository GitHub <https://github.com/your-username/AnalysingEnergy>`_
- `Documentation ReadTheDocs <https://analysingenergy.readthedocs.io>`_
- `Issues et bugs <https://github.com/your-username/AnalysingEnergy/issues>`_
- `Discussions <https://github.com/your-username/AnalysingEnergy/discussions>`_
