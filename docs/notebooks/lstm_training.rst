Notebook d'entraînement LSTM
===========================

Ce notebook détaille le processus complet d'entraînement des modèles LSTM pour la prédiction de production d'énergie verte.

Vue d'ensemble
--------------

Le notebook ``LSTM complet.ipynb`` couvre l'architecture, l'entraînement et l'optimisation des modèles LSTM pour prédire la production énergétique basée sur les données météorologiques et de demande.

Objectifs du notebook
---------------------

1. **Conception de l'architecture LSTM**
2. **Configuration des hyperparamètres**
3. **Entraînement avec validation croisée**
4. **Monitoring des performances**
5. **Sauvegarde des modèles optimaux**
6. **Évaluation et analyse des résultats**

Structure du notebook
---------------------

Imports et configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    import warnings
    warnings.filterwarnings('ignore')

Chargement des données préparées
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Chargement des données prétraitées
    X_train = np.load('Data/X_train.npy')
    y_train = np.load('Data/y_train.npy')
    X_val = np.load('Data/X_val.npy')
    y_val = np.load('Data/y_val.npy')
    X_test = np.load('Data/X_test.npy')
    y_test = np.load('Data/y_test.npy')
    
    print(f"Données d'entraînement: {X_train.shape}")
    print(f"Données de validation: {X_val.shape}")
    print(f"Données de test: {X_test.shape}")

Architecture du modèle LSTM
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Modèle de base :**

.. code-block:: python

    def create_lstm_model(input_shape, lstm_units=[50, 50], dropout_rate=0.2, 
                         learning_rate=0.001, l1_reg=0.01, l2_reg=0.01):
        """Crée un modèle LSTM multi-couches avec régularisation."""
        
        model = Sequential()
        
        # Première couche LSTM
        model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Couches LSTM additionnelles
        for i, units in enumerate(lstm_units[1:]):
            return_seq = i < len(lstm_units) - 2
            model.add(LSTM(
                units=units,
                return_sequences=return_seq,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Couches denses finales
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(dropout_rate/2))
        model.add(Dense(1, activation='linear'))
        
        # Compilation
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model

**Architecture avancée :**

.. code-block:: python

    def create_advanced_lstm_model(input_shape):
        """Modèle LSTM avancé avec architecture optimisée."""
        
        model = Sequential([
            # Première couche LSTM bidirectionnelle
            tf.keras.layers.Bidirectional(LSTM(
                64, return_sequences=True, 
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            ), input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # Deuxième couche LSTM
            LSTM(32, return_sequences=True,
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Troisième couche LSTM
            LSTM(16, return_sequences=False,
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Couches denses avec connexions résiduelles
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # Plus robuste aux outliers
            metrics=['mae', 'mse']
        )
        
        return model

Configuration des callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def setup_callbacks(model_name='lstm_model'):
        """Configure les callbacks pour l'entraînement."""
        
        callbacks = [
            # Arrêt précoce
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Sauvegarde du meilleur modèle
            ModelCheckpoint(
                filepath=f'Notebooks/models/{model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Réduction du learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks

Entraînement du modèle
~~~~~~~~~~~~~~~~~~~~~~

**Entraînement principal :**

.. code-block:: python

    # Création du modèle
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    
    # Affichage de l'architecture
    model.summary()
    
    # Configuration des callbacks
    callbacks = setup_callbacks('final_model')
    
    # Entraînement
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=False  # Important pour les séries temporelles
    )

**Entraînement avec validation croisée temporelle :**

.. code-block:: python

    def time_series_cv_training(X, y, n_splits=5):
        """Entraînement avec validation croisée temporelle."""
        
        cv_scores = []
        models = []
        
        # Division temporelle
        total_samples = len(X)
        split_size = total_samples // n_splits
        
        for i in range(n_splits):
            print(f"\n--- Fold {i+1}/{n_splits} ---")
            
            # Définition des indices
            train_end = (i + 1) * split_size
            val_start = train_end
            val_end = min(val_start + split_size // 2, total_samples)
            
            # Division des données
            X_fold_train = X[:train_end]
            y_fold_train = y[:train_end]
            X_fold_val = X[val_start:val_end]
            y_fold_val = y[val_start:val_end]
            
            # Création et entraînement du modèle
            model = create_lstm_model(input_shape)
            callbacks = setup_callbacks(f'model_fold_{i+1}')
            
            history = model.fit(
                X_fold_train, y_fold_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_fold_val, y_fold_val),
                callbacks=callbacks,
                verbose=0
            )
            
            # Évaluation
            val_loss = min(history.history['val_loss'])
            cv_scores.append(val_loss)
            models.append(model)
            
            print(f"Validation Loss: {val_loss:.4f}")
        
        return models, cv_scores

Monitoring et visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Visualisation de l'entraînement :**

.. code-block:: python

    def plot_training_history(history):
        """Visualise l'historique d'entraînement."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Évolution de la Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(history.history['mae'], label='Train MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Évolution de la MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate (si disponible)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Évolution du Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Gradient norms (si disponible)
        axes[1, 1].text(0.5, 0.5, 'Métriques additionnelles\n(gradient norms, etc.)',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Métriques Avancées')
        
        plt.tight_layout()
        plt.show()

**Monitoring en temps réel :**

.. code-block:: python

    class TrainingMonitor(tf.keras.callbacks.Callback):
        """Callback personnalisé pour le monitoring."""
        
        def __init__(self):
            self.losses = []
            self.val_losses = []
            self.best_val_loss = float('inf')
        
        def on_epoch_end(self, epoch, logs=None):
            self.losses.append(logs['loss'])
            self.val_losses.append(logs['val_loss'])
            
            if logs['val_loss'] < self.best_val_loss:
                self.best_val_loss = logs['val_loss']
                print(f"\n🎯 Nouveau meilleur modèle à l'epoch {epoch+1}: "
                      f"val_loss = {logs['val_loss']:.4f}")

Évaluation des performances
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Métriques d'évaluation :**

.. code-block:: python

    def evaluate_model(model, X_test, y_test, scaler_y):
        """Évalue le modèle sur les données de test."""
        
        # Prédictions
        y_pred_scaled = model.predict(X_test)
        
        # Dénormalisation
        y_test_original = scaler_y.inverse_transform(y_test)
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
        
        # Calcul des métriques
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        # Métriques relatives
        mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
        
        print("📊 Métriques de performance:")
        print(f"  RMSE: {rmse:.2f} MW")
        print(f"  MAE:  {mae:.2f} MW")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return {
            'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape,
            'y_true': y_test_original, 'y_pred': y_pred_original
        }

**Analyse des résidus :**

.. code-block:: python

    def analyze_residuals(y_true, y_pred):
        """Analyse des résidus de prédiction."""
        
        residuals = y_true.flatten() - y_pred.flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution des résidus
        axes[0, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution des Résidus')
        axes[0, 0].set_xlabel('Résidus (MW)')
        axes[0, 0].set_ylabel('Fréquence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot des Résidus')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Résidus vs prédictions
        axes[1, 0].scatter(y_pred.flatten(), residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Résidus vs Prédictions')
        axes[1, 0].set_xlabel('Prédictions (MW)')
        axes[1, 0].set_ylabel('Résidus (MW)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Résidus dans le temps
        axes[1, 1].plot(residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Résidus dans le Temps')
        axes[1, 1].set_xlabel('Index Temporel')
        axes[1, 1].set_ylabel('Résidus (MW)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

Optimisation des hyperparamètres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Grid Search pour LSTM :**

.. code-block:: python

    def lstm_grid_search():
        """Recherche par grille des meilleurs hyperparamètres."""
        
        param_grid = {
            'lstm_units': [[32, 16], [50, 25], [64, 32], [128, 64]],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64]
        }
        
        best_score = float('inf')
        best_params = None
        results = []
        
        import itertools
        
        # Toutes les combinaisons possibles
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            print(f"Testing: {params}")
            
            # Création et entraînement du modèle
            model = create_lstm_model(
                input_shape,
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate']
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=30,  # Réduction pour le grid search
                batch_size=params['batch_size'],
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Évaluation
            val_loss = min(history.history['val_loss'])
            results.append({**params, 'val_loss': val_loss})
            
            if val_loss < best_score:
                best_score = val_loss
                best_params = params
                print(f"  Nouveau meilleur: {val_loss:.4f}")
        
        return best_params, results

Sauvegarde et export
~~~~~~~~~~~~~~~~~~~

**Sauvegarde du modèle final :**

.. code-block:: python

    # Sauvegarde du modèle complet
    final_model_path = 'Notebooks/models/final_model.h5'
    model.save(final_model_path)
    
    # Sauvegarde avec métadonnées
    model_info = {
        'architecture': 'LSTM multi-couches',
        'input_shape': input_shape,
        'final_val_loss': min(history.history['val_loss']),
        'training_epochs': len(history.history['loss']),
        'hyperparameters': {
            'lstm_units': [50, 25],
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }
    }
    
    import json
    with open('Notebooks/models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

**Export des poids et architecture :**

.. code-block:: python

    # Sauvegarde de l'architecture seule
    model_json = model.to_json()
    with open('Notebooks/models/model_architecture.json', 'w') as f:
        f.write(model_json)
    
    # Sauvegarde des poids seuls
    model.save_weights('Notebooks/models/model_weights.h5')

Analyse comparative des modèles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Comparaison de différentes architectures :**

.. code-block:: python

    def compare_model_architectures():
        """Compare différentes architectures LSTM."""
        
        architectures = {
            'Simple LSTM': lambda: create_lstm_model(input_shape, [32]),
            'Deep LSTM': lambda: create_lstm_model(input_shape, [64, 32, 16]),
            'Wide LSTM': lambda: create_lstm_model(input_shape, [128, 64]),
            'Advanced LSTM': lambda: create_advanced_lstm_model(input_shape)
        }
        
        results = {}
        
        for name, model_func in architectures.items():
            print(f"\n🔧 Entraînement: {name}")
            
            model = model_func()
            callbacks = setup_callbacks(f'comparison_{name.lower().replace(" ", "_")}')
            
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            # Évaluation
            scaler_y = joblib.load('Notebooks/scalers/y_train_scaler.pkl')
            metrics = evaluate_model(model, X_test, y_test, scaler_y)
            
            results[name] = {
                'val_loss': min(history.history['val_loss']),
                'test_rmse': metrics['rmse'],
                'test_r2': metrics['r2'],
                'parameters': model.count_params()
            }
            
            print(f"  Val Loss: {results[name]['val_loss']:.4f}")
            print(f"  Test RMSE: {results[name]['test_rmse']:.2f}")
            print(f"  Parameters: {results[name]['parameters']:,}")
        
        return results

Utilisation pratique
-------------------

**Exécution du notebook :**

1. Assurez-vous que les données prétraitées sont disponibles
2. Exécutez les cellules d'architecture et configuration
3. Lancez l'entraînement avec monitoring
4. Évaluez les performances sur les données de test

**Personnalisation :**

- Modifiez l'architecture selon vos besoins de complexité
- Ajustez les hyperparamètres via grid search
- Adaptez les callbacks selon votre stratégie d'entraînement

**Optimisation des performances :**

- Utilisez GPU si disponible : ``tf.config.experimental.set_visible_devices``
- Optimisez les batch sizes selon votre mémoire
- Implémentez l'entraînement distribué pour les gros datasets

Prochaines étapes
----------------

Après l'entraînement avec ce notebook :

1. Consultez :doc:`prediction` pour utiliser le modèle entraîné
2. Explorez :doc:`../model_evaluation` pour une évaluation approfondie
3. Visitez :doc:`../hyperparameter_optimization` pour l'optimisation avancée

Troubleshooting
---------------

**Problèmes d'entraînement :**

- **Overfitting** : Augmentez le dropout, ajoutez de la régularisation
- **Underfitting** : Augmentez la complexité du modèle, réduisez la régularisation
- **Convergence lente** : Ajustez le learning rate, vérifiez les données

Pour plus d'aide, consultez :doc:`../troubleshooting`.
