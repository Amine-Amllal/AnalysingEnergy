Modèles LSTM
============

Cette section présente l'architecture et l'implémentation des modèles LSTM (Long Short-Term Memory) utilisés pour prédire la production d'énergie verte.

Introduction aux LSTM
---------------------

Pourquoi les LSTM ?
~~~~~~~~~~~~~~~~~~

Les réseaux LSTM sont particulièrement adaptés aux prédictions de séries temporelles car ils :

- **Mémorisent les dépendances long terme** : Crucial pour les patterns énergétiques saisonniers
- **Évitent le gradient vanishing** : Problème courant avec les RNN classiques  
- **Gèrent les séquences variables** : Flexibilité pour différents horizons de prédiction
- **Capturent les non-linéarités** : Importantes dans les systèmes énergétiques complexes

Architecture du modèle principal
---------------------------------

Modèle optimisé
~~~~~~~~~~~~~~

L'architecture finale, optimisée avec Optuna, présente la structure suivante :

.. code-block:: python

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    def build_optimized_model():
        model = Sequential([
            # Première couche LSTM avec return_sequences=True
            LSTM(74, activation='relu', return_sequences=True, 
                 input_shape=(1, 9)),
            
            # Deuxième couche LSTM  
            LSTM(69, activation='relu', return_sequences=False),
            
            # Couche de régularisation
            Dropout(0.1938),
            
            # Couche dense intermédiaire
            Dense(50, activation='relu'),
            
            # Couche de régularisation
            Dropout(0.1938),
            
            # Couche de sortie
            Dense(1)  # Régression : 1 neurone de sortie
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model

Paramètres clés
~~~~~~~~~~~~~~

.. list-table:: Hyperparamètres optimisés
   :widths: 25 20 55
   :header-rows: 1

   * - Paramètre
     - Valeur
     - Justification
   * - ``units_1``
     - 74
     - Première couche LSTM - capture les patterns complexes
   * - ``units_2``  
     - 69
     - Deuxième couche LSTM - extraction de features haut niveau
   * - ``dropout_rate``
     - 0.1938
     - Prévention du surapprentissage
   * - ``activation``
     - relu
     - Améliore la convergence vs tanh
   * - ``optimizer``
     - adam
     - Adaptation du learning rate automatique
   * - ``loss``
     - mse
     - Mean Squared Error pour la régression

Préparation des données pour LSTM
---------------------------------

Format d'entrée
~~~~~~~~~~~~~~

Les LSTM nécessitent une forme spécifique des données :

.. code-block:: python

    # Shape requis: (samples, timesteps, features)
    
    # Configuration actuelle
    n_input = 1        # 1 pas de temps (jour précédent)
    n_features = 9     # 9 variables d'entrée
    batch_size = 1     # Traitement séquentiel
    
    # Reshape des données
    X_reshaped = X.reshape((X.shape[0], n_input, n_features))
    
    print(f"Shape finale: {X_reshaped.shape}")
    # Output: (1515, 1, 9)

Générateur de séquences temporelles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utilisation du TimeseriesGenerator de Keras :

.. code-block:: python

    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    
    # Création du générateur
    generator = TimeseriesGenerator(
        data=X_train_scaled,     # Données d'entrée normalisées
        targets=y_train_scaled,  # Variable cible normalisée  
        length=n_input,          # Longueur des séquences
        batch_size=1            # Taille des lots
    )
    
    # Le générateur produit automatiquement les séquences
    for i, (x_batch, y_batch) in enumerate(generator):
        print(f"Batch {i}: X shape {x_batch.shape}, y shape {y_batch.shape}")
        if i == 2:  # Afficher seulement les 3 premiers
            break

Entraînement du modèle
----------------------

Configuration d'entraînement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Paramètres d'entraînement
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    # Entraînement du modèle
    history = model.fit(
        generator,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=validation_generator  # Si disponible
    )

Callbacks utiles
~~~~~~~~~~~~~~~

.. code-block:: python

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    callbacks = [
        # Arrêt précoce si pas d'amélioration
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        
        # Sauvegarde du meilleur modèle
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        
        # Réduction du learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    history = model.fit(generator, epochs=EPOCHS, callbacks=callbacks)

Surveillance de l'entraînement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    
    def plot_training_history(history):
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Métriques additionnelles si disponibles
        plt.subplot(1, 2, 2)
        if 'mae' in history.history:
            plt.plot(history.history['mae'], label='MAE')
        plt.title('Model Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

Prédictions et évaluation
-------------------------

Génération de prédictions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Prédictions sur les données de test
    y_pred_scaled = model.predict(X_test_reshaped)
    
    # Dénormalisation pour obtenir les valeurs réelles
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test_scaled)

Métriques d'évaluation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    
    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mae = mean_absolute_error(y_test_actual, y_pred)
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
    
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW") 
    print(f"MAPE: {mape:.2f}%")
    
    # Résultats typiques du modèle optimisé
    # RMSE: 291.19 MW
    # MAE:  235.47 MW
    # MAPE: 3.42%

Visualisation des résultats
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def plot_predictions(y_true, y_pred, dates):
        plt.figure(figsize=(15, 6))
        
        plt.plot(dates, y_true, label='Valeurs réelles', 
                color='blue', alpha=0.7)
        plt.plot(dates, y_pred, label='Prédictions', 
                color='red', alpha=0.7)
        
        plt.xlabel('Date')
        plt.ylabel('Production maximale (MW)')
        plt.title('Comparaison Prédictions vs Réalité')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

Modèles spécialisés par variable
--------------------------------

Architecture modulaire
~~~~~~~~~~~~~~~~~~~~~~

Le projet inclut des modèles LSTM séparés pour chaque variable :

.. code-block:: python

    models_dict = {
        'temp2_ave(c)': 'temp2_ave(c)_LSTM.h5',
        'temp2_max(c)': 'temp2_max(c)_LSTM.h5', 
        'temp2_min(c)': 'temp2_min(c)_LSTM.h5',
        'wind_speed50_ave(m/s)': 'wind_speed50_ave(ms)_LSTM.h5',
        'wind_speed50_max(m/s)': 'wind_speed50_max(ms)_LSTM.h5',
        'wind_speed50_min(m/s)': 'wind_speed50_min(ms)_LSTM.h5',
        'suface_pressure(pa)': 'suface_pressure(pa)_LSTM.h5',
        'prectotcorr': 'prectotcorr_LSTM.h5',
        'total_demand(mw)': 'total_demand(mw)_LSTM.h5'
    }

Utilisation des modèles spécialisés
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def load_specialized_model(variable_name):
        """Charge un modèle LSTM spécialisé pour une variable donnée"""
        
        # Charger le modèle
        model_path = f"models/{variable_name.replace('/', '')}_LSTM.h5"
        model = load_model(model_path)
        
        # Charger le scaler correspondant
        scaler_path = f"scalers/{variable_name.replace('/', '')}_scaler.pkl"
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    
    # Exemple d'utilisation
    temp_model, temp_scaler = load_specialized_model('temp2_ave(c)')

Ensemble de modèles
~~~~~~~~~~~~~~~~~~~

Combinaison de plusieurs modèles pour améliorer les prédictions :

.. code-block:: python

    def ensemble_prediction(X_test, models_dict):
        """Combine les prédictions de plusieurs modèles"""
        predictions = []
        
        for variable, model_path in models_dict.items():
            model = load_model(model_path)
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # Moyenne pondérée ou vote majoritaire
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

Optimisation avancée
-------------------

Transfer Learning
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Pré-entraînement sur une variable, fine-tuning sur une autre
    def transfer_learning_model(source_model_path, target_variable):
        # Charger le modèle pré-entraîné
        base_model = load_model(source_model_path)
        
        # Geler certaines couches
        for layer in base_model.layers[:-2]:  # Garder les dernières couches entraînables
            layer.trainable = False
        
        # Recompiler avec un learning rate plus faible
        base_model.compile(optimizer=Adam(lr=0.0001), loss='mse')
        
        return base_model

Régularisation avancée
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tensorflow.keras.regularizers import l1_l2
    
    def regularized_model():
        model = Sequential([
            LSTM(74, activation='relu', return_sequences=True,
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            LSTM(69, activation='relu'),
            Dropout(0.3),
            Dense(50, activation='relu',
                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.3),
            Dense(1)
        ])
        return model

Déploiement et sauvegarde
------------------------

Sauvegarde complète du modèle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Sauvegarde du modèle
    model.save('final_model.h5')
    
    # Sauvegarde des scalers
    joblib.dump(scaler_X, 'X_scaler.pkl')
    joblib.dump(scaler_y, 'y_scaler.pkl')
    
    # Sauvegarde des métadonnées
    import json
    
    model_info = {
        'architecture': 'LSTM',
        'input_shape': (1, 9),
        'units_1': 74,
        'units_2': 69,
        'dropout_rate': 0.1938,
        'rmse': 291.19,
        'training_date': '2025-06-01'
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f)

Chargement pour production
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def load_production_model():
        """Charge le modèle pour utilisation en production"""
        
        # Charger le modèle
        model = load_model('final_model.h5')
        
        # Charger les scalers
        scaler_X = joblib.load('X_scaler.pkl')
        scaler_y = joblib.load('y_scaler.pkl')
        
        # Charger les métadonnées
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        return model, scaler_X, scaler_y, model_info

Bonnes pratiques
---------------

Validation temporelle
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.model_selection import TimeSeriesSplit
    
    # Validation croisée temporelle
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_idx, val_idx in tscv.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Entraîner et évaluer le modèle
        # ...

Monitoring en production
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def monitor_model_drift(new_data, original_stats):
        """Surveille la dérive du modèle en production"""
        
        # Calculer les statistiques des nouvelles données
        new_stats = new_data.describe()
        
        # Comparer avec les statistiques d'origine
        drift_threshold = 0.1  # 10% de changement
        
        for column in new_stats.columns:
            original_mean = original_stats[column]['mean']
            new_mean = new_stats[column]['mean']
            
            drift = abs(new_mean - original_mean) / original_mean
            
            if drift > drift_threshold:
                print(f"Dérive détectée pour {column}: {drift:.2%}")

Prochaines étapes
----------------

Maintenant que vous maîtrisez les modèles LSTM :

1. Explorez :doc:`hyperparameter_optimization` pour optimiser davantage
2. Consultez :doc:`model_evaluation` pour l'évaluation approfondie  
3. Découvrez :doc:`notebooks/lstm_training` pour les exemples pratiques
