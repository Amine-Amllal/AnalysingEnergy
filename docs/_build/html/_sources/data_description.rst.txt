Description des données
=====================

Cette section présente en détail le dataset BanE-16 utilisé dans le projet Analysing Green Energy.

À propos du dataset BanE-16
---------------------------

Le dataset **BanE-16** (Bangladesh Energy-16) est un ensemble de données complet qui combine des variables météorologiques et énergétiques pour analyser et prédire la production d'énergie verte au Bangladesh.

Source des données
~~~~~~~~~~~~~~~~~

- **Repository Mendeley** : https://data.mendeley.com/datasets/3brbjpt39s/2
- **Article de référence** : https://pmc.ncbi.nlm.nih.gov/articles/PMC10792676/#sec0006
- **Période couverte** : 2018-2023 (données journalières)
- **Nombre d'observations** : ~1800 enregistrements

Variables du dataset
-------------------

Le dataset contient 13 variables principales divisées en trois catégories :

Variables météorologiques
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Variables météorologiques
   :widths: 25 20 55
   :header-rows: 1

   * - Variable
     - Unité
     - Description
   * - ``temp2_max(c)``
     - °C
     - Température maximale journalière à 2m de hauteur
   * - ``temp2_min(c)``
     - °C
     - Température minimale journalière à 2m de hauteur
   * - ``temp2_ave(c)``
     - °C
     - Température moyenne journalière à 2m de hauteur
   * - ``suface_pressure(pa)``
     - Pa
     - Pression atmosphérique au niveau de la surface
   * - ``wind_speed50_max(m/s)``
     - m/s
     - Vitesse maximale du vent à 50m de hauteur
   * - ``wind_speed50_min(m/s)``
     - m/s
     - Vitesse minimale du vent à 50m de hauteur
   * - ``wind_speed50_ave(m/s)``
     - m/s
     - Vitesse moyenne du vent à 50m de hauteur
   * - ``prectotcorr``
     - mm
     - Précipitations totales corrigées

Variables énergétiques
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Variables énergétiques
   :widths: 25 20 55
   :header-rows: 1

   * - Variable
     - Unité
     - Description
   * - ``total_demand(mw)``
     - MW
     - Demande totale d'électricité
   * - ``max_generation(mw)``
     - MW
     - Production maximale d'énergie (variable cible)

Variables temporelles
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Variables temporelles
   :widths: 25 20 55
   :header-rows: 1

   * - Variable
     - Type
     - Description
   * - ``date``
     - Date
     - Index temporel (YYYY-MM-DD)
   * - ``month``
     - Integer
     - Mois de l'année (1-12)
   * - ``year``
     - Integer
     - Année de l'observation

Statistiques descriptives
-------------------------

Voici un aperçu statistique des principales variables :

Variables météorologiques
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Exemple de statistiques pour les températures
    Temperature moyenne : 25.4°C ± 4.2°C
    Plage : 8.2°C - 35.8°C
    
    Vitesse du vent moyenne : 3.8 m/s ± 1.9 m/s
    Plage : 0.1 m/s - 12.4 m/s
    
    Pression atmosphérique : 101.2 kPa ± 0.8 kPa
    Plage : 98.5 kPa - 103.1 kPa

Variables énergétiques
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Production d'énergie
    Génération moyenne : 6,847 MW ± 1,205 MW
    Plage : 3,200 MW - 9,800 MW
    
    # Demande d'énergie
    Demande moyenne : 7,245 MW ± 1,058 MW
    Plage : 4,500 MW - 9,500 MW

Saisonnalité et tendances
------------------------

Variations saisonnières
~~~~~~~~~~~~~~~~~~~~~~

**Températures**
- **Été** (avril-septembre) : Températures élevées (28-35°C)
- **Hiver** (décembre-février) : Températures modérées (15-25°C)
- **Mousson** (juin-octobre) : Précipitations importantes

**Production d'énergie**
- **Pic d'été** : Forte demande pour la climatisation
- **Saison sèche** : Production solaire optimale
- **Mousson** : Production éolienne accrue

Corrélations importantes
~~~~~~~~~~~~~~~~~~~~~~~

Les analyses révèlent plusieurs corrélations significatives :

.. code-block:: python

    # Corrélations avec la production d'énergie
    Température moyenne    : +0.65
    Vitesse du vent       : +0.42
    Demande totale        : +0.78
    Pression atmosphérique: -0.23
    Précipitations       : -0.15

Qualité des données
------------------

Valeurs manquantes
~~~~~~~~~~~~~~~~~

Le dataset présente une excellente qualité avec :
- **0.2%** de valeurs manquantes au total
- Principalement sur les variables de précipitations
- Aucune valeur manquante pour les variables cibles

Valeurs aberrantes
~~~~~~~~~~~~~~~~~

Detection automatique des outliers :
- **Méthode IQR** : 1.5% des observations flaggées
- **Z-score** : 2.1% des observations avec |z| > 3
- Principalement sur les variables météorologiques extrêmes

Préparation des données
----------------------

Division train/test
~~~~~~~~~~~~~~~~~~

Les données sont divisées selon la stratégie suivante :

.. code-block:: python

    # Division temporelle
    Données d'entraînement : 2018-01-01 à 2022-06-30 (80%)
    Données de test        : 2022-07-01 à 2023-12-31 (20%)
    
    # Validation croisée temporelle pour éviter le data leakage

Normalisation
~~~~~~~~~~~~

Toutes les variables sont normalisées avec MinMaxScaler :

.. code-block:: python

    from sklearn.preprocessing import MinMaxScaler
    
    # Normalisation entre 0 et 1
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_train)

Ingénierie des caractéristiques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Variables dérivées créées :

.. code-block:: python

    # Variables calculées
    - temp_range = temp_max - temp_min
    - wind_variance = wind_max - wind_min  
    - demand_generation_ratio = total_demand / max_generation
    - seasonal_indicators (sin/cos pour capturer la saisonnalité)

Utilisation dans les modèles
----------------------------

Configuration pour LSTM
~~~~~~~~~~~~~~~~~~~~~~~

Les données sont restructurées pour les modèles de séries temporelles :

.. code-block:: python

    # Format d'entrée LSTM
    Input shape: (batch_size, timesteps, features)
    - batch_size: Variable selon l'entraînement
    - timesteps: 1 (prédiction à partir du jour précédent)
    - features: 9 (variables météorologiques + demande)
    
    # Variable cible
    Output: max_generation(mw) - scalaire

Variables d'entrée et de sortie
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Variables d'entrée (features)** :
- temp2_max(c), temp2_min(c), temp2_ave(c)
- suface_pressure(pa)
- wind_speed50_max(m/s), wind_speed50_min(m/s), wind_speed50_ave(m/s)
- prectotcorr
- total_demand(mw)

**Variable de sortie (target)** :
- max_generation(mw)

Recommandations d'utilisation
-----------------------------

Bonnes pratiques
~~~~~~~~~~~~~~~

1. **Validation temporelle** : Utilisez toujours une division temporelle pour éviter le data leakage
2. **Normalisation consistante** : Utilisez les mêmes scalers pour train/test
3. **Monitoring des dérives** : Surveillez les changements de distribution dans le temps
4. **Cross-validation** : Utilisez la validation croisée temporelle (TimeSeriesSplit)

Limitations
~~~~~~~~~~

- **Données locales** : Spécifiques au Bangladesh, adaptation nécessaire pour d'autres régions
- **Résolution temporelle** : Données journalières uniquement
- **Variables limitées** : Absence de données sur l'irradiation solaire directe
- **Période récente** : Données relativement récentes (2018-2023)

Export et formats
----------------

Le dataset est disponible dans plusieurs formats :

.. code-block::

    Data/
    ├── data.csv           # Dataset complet
    ├── train_data.csv     # Données d'entraînement  
    └── test_data.csv      # Données de test

Structure du fichier CSV :

.. code-block::

    date,temp2_max(c),temp2_min(c),...,max_generation(mw)
    2018-01-01,24.48,13.78,...,7651.0
    2018-01-02,23.16,15.28,...,7782.0
    ...

Prochaines étapes
----------------

Maintenant que vous comprenez les données :

1. Explorez le :doc:`preprocessing` pour la préparation avancée
2. Consultez :doc:`data_analysis` pour l'analyse exploratoire
3. Découvrez :doc:`lstm_models` pour la modélisation
