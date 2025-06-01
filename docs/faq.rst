Questions Fréquentes (FAQ)
========================

Cette section répond aux questions les plus fréquemment posées concernant le projet AnalysingEnergy.

Questions générales
------------------

Qu'est-ce que le projet AnalysingEnergy ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le projet AnalysingEnergy est un système de prédiction de génération d'énergie verte utilisant des réseaux de neurones LSTM (Long Short-Term Memory). Il analyse les données météorologiques du dataset BanE-16 pour prédire la production énergétique quotidienne avec une précision élevée (RMSE: 291.19 MW).

**Caractéristiques principales :**

- Modèles LSTM optimisés avec Optuna
- Interface utilisateur Streamlit intuitive  
- Prédictions court et long terme
- Visualisations interactives
- Documentation complète

Quelle est la précision du modèle ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notre modèle LSTM optimisé atteint les performances suivantes :

- **RMSE** : 291.19 MW (8.9% de la capacité maximale)
- **MAE** : ~185 MW
- **R²** : 0.847
- **Corrélation** : 0.921

Ces métriques placent notre modèle bien au-dessus des méthodes de référence (persistence, moyenne mobile, etc.).

Puis-je utiliser le projet avec mes propres données ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Oui, le projet est conçu pour être adaptable. Vos données doivent contenir :

**Colonnes requises :**

- Variables météorologiques : température (min/mean/max), vitesse du vent (min/mean/max), précipitations, pression, humidité
- Variable cible : génération d'énergie

**Format :**

.. code-block:: python

   # Exemple de format attendu
   required_columns = [
       'min_temperature', 'mean_temperature', 'max_temperature',
       'min_windspeed', 'mean_windspeed', 'max_windspeed', 
       'total_precipitation', 'surface_pressure', 'mean_relative_humidity',
       'max_generation(mw)'  # Variable cible
   ]

Pour adapter vos données :

.. code-block:: python

   # Mapper vos colonnes vers le format attendu
   column_mapping = {
       'your_temp_min': 'min_temperature',
       'your_temp_avg': 'mean_temperature',
       'your_generation': 'max_generation(mw)',
       # ... autres mappings
   }
   
   data = data.rename(columns=column_mapping)

Questions techniques
-------------------

Quelles sont les dépendances requises ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dépendances principales :**

- Python 3.8+
- TensorFlow 2.13.0
- Pandas 1.5+
- NumPy 1.24+
- Scikit-learn 1.3+
- Streamlit 1.28+
- Optuna 3.4+
- Plotly 5.17+

**Installation complète :**

.. code-block:: bash

   pip install -r requirements.txt

Pour créer le fichier requirements.txt :

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

Comment optimiser les performances ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pour l'entraînement :**

1. **Utiliser un GPU** si disponible
   
   .. code-block:: python
   
      import tensorflow as tf
      print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")

2. **Optimiser les paramètres**
   
   .. code-block:: python
   
      # Paramètres optimaux trouvés
      optimal_params = {
           'units_1': 74,
           'units_2': 69, 
           'dropout_rate': 0.1938,
           'batch_size': 32,
           'learning_rate': 0.001
       }

3. **Pipeline de données efficace**
   
   .. code-block:: python
   
      # Utiliser tf.data pour de meilleures performances
      dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
      dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

**Pour les prédictions :**

- Prédire par batches plutôt qu'individuellement
- Utiliser des modèles quantifiés pour le déploiement
- Mettre en cache les prédictions fréquentes

Pourquoi utiliser LSTM plutôt que d'autres modèles ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Avantages des LSTM pour la prédiction énergétique :**

1. **Mémoire long terme** : Capture les patterns saisonniers et cycliques
2. **Gestion des séquences** : Idéal pour les séries temporelles
3. **Non-linéarité** : Modélise les relations complexes météo-énergie
4. **Robustesse** : Gère bien les valeurs manquantes et le bruit

**Comparaison avec d'autres approches :**

.. code-block:: text

   Modèle              RMSE (MW)    R²      Avantages
   ═══════════════════════════════════════════════════
   LSTM (optimisé)     291.19      0.847   Meilleure précision
   Random Forest       340.25      0.782   Interprétabilité  
   ARIMA              425.67      0.651   Simplicité
   Régression Linéaire 580.12      0.423   Rapidité
   Persistence        612.45      0.385   Baseline simple

Questions sur l'utilisation
---------------------------

Comment faire une prédiction simple ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Via l'interface Streamlit :**

1. Lancer l'application : `streamlit run interface/app.py`
2. Entrer les paramètres météorologiques
3. Cliquer sur "Prédire"

**Via code Python :**

.. code-block:: python

   from interface.app import EnergyPredictor
   
   # Initialisation
   predictor = EnergyPredictor()
   predictor.load_trained_models()
   
   # Prédiction
   prediction = predictor.predict_single_day(
       min_temp=18.0, mean_temp=25.0, max_temp=32.0,
       wind_speed=15.0, precipitation=0.0,
       pressure=1013.0, humidity=60.0
   )
   
   print(f"Génération prédite: {prediction:.2f} MW")

Comment interpréter les résultats ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Unités et échelles :**

- Prédictions en **mégawatts (MW)**
- Valeurs typiques : 0-5000 MW selon la capacité installée
- Intervalle de confiance à considérer (±291 MW en moyenne)

**Facteurs d'influence :**

1. **Vitesse du vent** : Impact le plus important (corrélation ~0.75)
2. **Température** : Impact modéré (corrélation ~0.45)  
3. **Saison** : Variations saisonnières marquées
4. **Conditions météo extrêmes** : Peuvent causer des écarts

**Exemple d'interprétation :**

.. code-block:: text

   Prédiction: 1250 MW
   
   Interprétation:
   - Production élevée (> moyenne de 800 MW)
   - Conditions favorables (vent fort, température modérée)
   - Confiance élevée (conditions dans la plage d'entraînement)
   - Recommandation: Planifier pour forte production

Comment entraîner un nouveau modèle ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Entraînement avec les données existantes :**

.. code-block:: python

   predictor = EnergyPredictor()
   predictor.load_data('Data/data.csv')
   
   # Entraînement avec paramètres par défaut
   history = predictor.train_models(epochs=100, batch_size=32)
   
   # Évaluation
   metrics = predictor.evaluate_models()
   print(f"RMSE: {metrics['RMSE']:.2f}")

**Entraînement avec optimisation :**

.. code-block:: python

   # Optimisation des hyperparamètres (plus long)
   from your_module import ModelOptimizer
   
   optimizer = ModelOptimizer(X_train, y_train, X_val, y_val)
   best_params = optimizer.optimize(n_trials=50)
   
   # Utilisation des meilleurs paramètres
   predictor.train_models(**best_params)

**Temps d'entraînement typiques :**

- CPU : 2-4 heures (selon les paramètres)
- GPU : 30-60 minutes
- Optimisation : 8-12 heures (50 trials)

Questions sur les données
------------------------

Quelle est la qualité requise des données ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critères minimaux :**

- **Complétude** : < 10% de valeurs manquantes par variable
- **Cohérence** : Valeurs dans des plages réalistes
- **Fréquence** : Données quotidiennes (minimum)
- **Durée** : Au moins 2 ans pour capturer la saisonnalité

**Vérification de qualité :**

.. code-block:: python

   def check_data_quality(data):
       """Vérification de la qualité des données"""
       
       report = {}
       
       # Valeurs manquantes
       missing_pct = (data.isnull().sum() / len(data)) * 100
       report['missing_data'] = missing_pct[missing_pct > 0].to_dict()
       
       # Valeurs aberrantes (méthode IQR)
       numeric_cols = data.select_dtypes(include=[np.number]).columns
       outliers = {}
       for col in numeric_cols:
           Q1, Q3 = data[col].quantile([0.25, 0.75])
           IQR = Q3 - Q1
           outlier_count = ((data[col] < Q1 - 1.5*IQR) | 
                           (data[col] > Q3 + 1.5*IQR)).sum()
           outliers[col] = outlier_count
       report['outliers'] = outliers
       
       # Plages de valeurs
       report['value_ranges'] = data.describe().to_dict()
       
       return report

Comment gérer les valeurs manquantes ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Stratégies selon le pourcentage :**

- **< 5%** : Interpolation linéaire
- **5-15%** : Méthodes avancées (KNN, régression)
- **> 15%** : Analyse de la cause, possible exclusion

**Implémentation :**

.. code-block:: python

   # Interpolation simple
   data['temperature'] = data['temperature'].interpolate(method='linear')
   
   # Imputation KNN pour patterns complexes
   from sklearn.impute import KNNImputer
   
   imputer = KNNImputer(n_neighbors=5)
   data_imputed = imputer.fit_transform(data[numeric_columns])

Puis-je utiliser des données horaires ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le modèle actuel est optimisé pour des données quotidiennes. Pour des données horaires :

**Adaptations nécessaires :**

1. **Agrégation** vers le quotidien
   
   .. code-block:: python
   
      # Agrégation quotidienne
      daily_data = hourly_data.resample('D').agg({
           'temperature': 'mean',
           'wind_speed': 'mean', 
           'generation': 'max',  # Pic de génération
           'precipitation': 'sum'
       })

2. **Modèle haute fréquence** (développement futur)
   
   - Séquences plus courtes (24h au lieu de 60 jours)
   - Architecture adaptée
   - Plus de données d'entraînement requises

Questions sur le déploiement
---------------------------

Comment déployer l'application en production ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Options de déploiement :**

1. **Serveur local**
   
   .. code-block:: bash
   
      # Production locale
      streamlit run interface/app.py --server.port 8501 --server.address 0.0.0.0

2. **Cloud (Heroku, AWS, etc.)**
   
   Créer un `Procfile` :
   
   .. code-block:: text
   
      web: streamlit run interface/app.py --server.port=$PORT --server.address=0.0.0.0

3. **Conteneurisation Docker**
   
   .. code-block:: dockerfile
   
      FROM python:3.9-slim
      
      WORKDIR /app
      COPY requirements.txt .
      RUN pip install -r requirements.txt
      
      COPY . .
      
      EXPOSE 8501
      CMD ["streamlit", "run", "interface/app.py"]

Comment intégrer via API REST ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Création d'une API Flask :**

.. code-block:: python

   from flask import Flask, request, jsonify
   from interface.app import EnergyPredictor
   
   app = Flask(__name__)
   predictor = EnergyPredictor()
   predictor.load_trained_models()
   
   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.json
       
       prediction = predictor.predict_single_day(
           min_temp=data['min_temp'],
           mean_temp=data['mean_temp'],
           max_temp=data['max_temp'],
           wind_speed=data['wind_speed'],
           precipitation=data['precipitation'],
           pressure=data['pressure'],
           humidity=data['humidity']
       )
       
       return jsonify({'prediction': float(prediction)})
   
   if __name__ == '__main__':
       app.run(debug=False, host='0.0.0.0', port=5000)

**Utilisation de l'API :**

.. code-block:: python

   import requests
   
   # Appel à l'API
   response = requests.post('http://localhost:5000/predict', json={
       'min_temp': 18.0,
       'mean_temp': 25.0,
       'max_temp': 32.0,
       'wind_speed': 15.0,
       'precipitation': 0.0,
       'pressure': 1013.0,
       'humidity': 60.0
   })
   
   prediction = response.json()['prediction']
   print(f"Prédiction: {prediction:.2f} MW")

Quelles sont les limites du modèle ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Limites techniques :**

1. **Horizon de prédiction** : Précision décroît au-delà de 30 jours
2. **Conditions extrêmes** : Performance réduite lors d'événements exceptionnels
3. **Généralisation** : Optimisé pour le dataset BanE-16 spécifique
4. **Variables d'entrée** : Limité aux variables météorologiques disponibles

**Limites pratiques :**

- Nécessite des prévisions météorologiques fiables
- Performance dépendante de la qualité des données d'entraînement
- Réentraînement périodique recommandé

**Améliorations futures :**

- Intégration de données satellite
- Modèles ensemble pour réduire l'incertitude
- Variables économiques et réglementaires
- Prédictions probabilistes avec intervalles de confiance

Questions de maintenance
-----------------------

À quelle fréquence réentraîner le modèle ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommandations :**

- **Mensuellement** : Ajout des nouvelles données
- **Trimestriellement** : Réévaluation complète des performances
- **Annuellement** : Optimisation des hyperparamètres

**Indicateurs de dégradation :**

.. code-block:: python

   def monitor_model_performance(current_rmse, baseline_rmse=291.19):
       """Surveillance de la performance du modèle"""
       
       degradation = (current_rmse - baseline_rmse) / baseline_rmse * 100
       
       if degradation > 20:
           return "Réentraînement urgent requis"
       elif degradation > 10:
           return "Réentraînement recommandé"
       else:
           return "Performance acceptable"

Comment sauvegarder et restaurer les modèles ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Sauvegarde complète :**

.. code-block:: python

   import pickle
   from datetime import datetime
   
   # Sauvegarde du modèle avec métadonnées
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
   # Modèle TensorFlow
   model.save(f'models/lstm_model_{timestamp}.h5')
   
   # Scalers
   with open(f'scalers/scaler_{timestamp}.pkl', 'wb') as f:
       pickle.dump(scaler, f)
   
   # Métadonnées
   metadata = {
       'rmse': 291.19,
       'training_date': timestamp,
       'hyperparameters': optimal_params,
       'data_version': 'v1.0'
   }
   
   with open(f'models/metadata_{timestamp}.json', 'w') as f:
       json.dump(metadata, f)

**Restauration :**

.. code-block:: python

   # Chargement complet
   model = tf.keras.models.load_model('models/lstm_model_20231201_143022.h5')
   
   with open('scalers/scaler_20231201_143022.pkl', 'rb') as f:
       scaler = pickle.load(f)

Support et communauté
--------------------

Où obtenir de l'aide ?
~~~~~~~~~~~~~~~~~~~~~~

1. **Documentation** : Consultez d'abord cette documentation complète
2. **Troubleshooting** : :doc:`troubleshooting` pour les problèmes courants
3. **GitHub Issues** : Pour rapporter des bugs ou demander des fonctionnalités
4. **Email** : Contact direct avec l'équipe de développement

Comment contribuer au projet ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Types de contributions :**

- Rapports de bugs
- Suggestions d'améliorations
- Nouvelles fonctionnalités
- Documentation
- Tests et validation

**Processus de contribution :**

1. Fork du repository
2. Création d'une branche feature
3. Développement et tests
4. Pull request avec description détaillée

Ressources supplémentaires
-------------------------

- **Documentation API** : :doc:`api_reference`
- **Notebooks exemples** : :doc:`notebooks/index`
- **Guide d'optimisation** : :doc:`hyperparameter_optimization`
- **Analyse des données** : :doc:`data_analysis`

.. note::

   Cette FAQ est mise à jour régulièrement. N'hésitez pas à suggérer de nouvelles questions qui pourraient aider la communauté.
