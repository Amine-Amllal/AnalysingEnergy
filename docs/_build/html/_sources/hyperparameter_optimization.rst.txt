Optimisation des hyperparamètres
=================================

Cette section détaille l'utilisation d'Optuna pour optimiser automatiquement les hyperparamètres des modèles LSTM.

Introduction à Optuna
---------------------

Pourquoi Optuna ?
~~~~~~~~~~~~~~~~

Optuna est une bibliothèque d'optimisation bayésienne qui présente plusieurs avantages :

- **Optimisation intelligente** : Utilise des algorithmes sophistiqués (TPE, CMA-ES)
- **Parallélisation** : Support natif pour l'optimisation distribuée
- **Pruning automatique** : Arrête les essais non prometteurs rapidement
- **Interface simple** : API intuitive et flexible
- **Visualisations** : Outils intégrés pour analyser les résultats

Configuration de base
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    
    # Configuration du sampler et pruner
    sampler = TPESampler(seed=42)  # Reproductibilité
    pruner = MedianPruner(
        n_startup_trials=5,    # Essais avant activation du pruning
        n_warmup_steps=10,     # Étapes avant évaluation
        interval_steps=5       # Fréquence d'évaluation
    )

Définition de l'espace de recherche
-----------------------------------

Fonction objective
~~~~~~~~~~~~~~~~~

La fonction objective définit ce que nous voulons optimiser :

.. code-block:: python

    def objective(trial):
        """
        Fonction objective pour Optuna.
        Retourne la métrique à minimiser (validation loss).
        """
        
        # 1. Suggestion des hyperparamètres
        units_1 = trial.suggest_int('units_1', 50, 150)
        units_2 = trial.suggest_int('units_2', 50, 100)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # 2. Construction du modèle avec les hyperparamètres suggérés
        model = build_model(
            units_1=units_1,
            units_2=units_2, 
            dropout_rate=dropout_rate,
            activation=activation,
            learning_rate=learning_rate
        )
        
        # 3. Entraînement avec early stopping
        history = model.fit(
            train_generator,
            epochs=50,
            validation_data=val_generator,
            verbose=0,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # 4. Évaluation sur validation
        val_loss = min(history.history['val_loss'])
        
        return val_loss

Espace de recherche détaillé
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Hyperparamètres optimisés
   :widths: 25 30 20 25
   :header-rows: 1

   * - Paramètre
     - Plage de recherche
     - Type
     - Impact
   * - ``units_1``
     - 50-150
     - Integer
     - Capacité de la 1ère couche LSTM
   * - ``units_2``
     - 50-100  
     - Integer
     - Capacité de la 2ème couche LSTM
   * - ``dropout_rate``
     - 0.1-0.5
     - Float
     - Régularisation
   * - ``activation``
     - ['tanh', 'relu']
     - Categorical
     - Fonction d'activation
   * - ``learning_rate``
     - 1e-4 à 1e-2
     - Float (log)
     - Vitesse d'apprentissage
   * - ``batch_size``
     - [16, 32, 64]
     - Categorical
     - Taille des lots
   * - ``epochs``
     - 30-100
     - Integer
     - Nombre d'époques max

Construction du modèle optimisable
----------------------------------

Fonction de construction modulaire
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def build_model(units_1=100, units_2=50, dropout_rate=0.2, 
                   activation='tanh', learning_rate=0.001):
        """
        Construit un modèle LSTM avec des hyperparamètres configurables.
        """
        
        model = Sequential([
            LSTM(units_1, 
                 activation=activation, 
                 return_sequences=True, 
                 input_shape=(n_input, n_features)),
            
            LSTM(units_2, 
                 activation=activation, 
                 return_sequences=False),
            
            Dropout(dropout_rate),
            
            Dense(50, activation='relu'),
            
            Dropout(dropout_rate),
            
            Dense(1)  # Sortie pour régression
        ])
        
        # Optimiseur avec learning rate configurable
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model

Validation avec pruning
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def objective_with_pruning(trial):
        """
        Fonction objective avec pruning pour arrêter les essais non prometteurs.
        """
        
        # Paramètres suggérés
        params = {
            'units_1': trial.suggest_int('units_1', 50, 150),
            'units_2': trial.suggest_int('units_2', 50, 100),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'activation': trial.suggest_categorical('activation', ['tanh', 'relu'])
        }
        
        model = build_model(**params)
        
        # Entraînement avec rapports intermédiaires pour pruning
        for epoch in range(50):
            history = model.fit(
                train_generator,
                epochs=1,
                validation_data=val_generator,
                verbose=0
            )
            
            # Rapport de la loss de validation à Optuna
            val_loss = history.history['val_loss'][0]
            trial.report(val_loss, epoch)
            
            # Vérification si l'essai doit être prunée
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return val_loss

Lancement de l'optimisation
---------------------------

Configuration de l'étude
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def run_optimization(n_trials=100):
        """
        Lance l'optimisation Optuna avec configuration complète.
        """
        
        # Création de l'étude
        study = optuna.create_study(
            direction='minimize',           # Minimiser la validation loss
            sampler=TPESampler(seed=42),   # Algorithme TPE
            pruner=MedianPruner(           # Pruning médian
                n_startup_trials=10,
                n_warmup_steps=15,
                interval_steps=5
            ),
            study_name='lstm_optimization',
            storage='sqlite:///optuna_study.db',  # Persistance
            load_if_exists=True            # Reprendre si existe
        )
        
        # Optimisation
        study.optimize(
            objective_with_pruning, 
            n_trials=n_trials,
            timeout=3600,  # 1 heure maximum
            callbacks=[logging_callback]  # Logging personnalisé
        )
        
        return study

Callbacks et monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def logging_callback(study, trial):
        """Callback pour logger les progrès de l'optimisation."""
        
        print(f"Trial {trial.number} terminé:")
        print(f"  Valeur: {trial.value:.4f}")
        print(f"  Paramètres: {trial.params}")
        print(f"  État: {trial.state}")
        
        if trial.number % 10 == 0:
            print(f"\nMeilleur essai jusqu'à présent:")
            print(f"  Numéro: {study.best_trial.number}")
            print(f"  Valeur: {study.best_value:.4f}")
            print(f"  Paramètres: {study.best_params}")

Résultats de l'optimisation
---------------------------

Meilleurs paramètres trouvés
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Après optimisation avec 50 essais, les meilleurs hyperparamètres obtenus sont :

.. code-block:: python

    best_params = {
        'units_1': 74,
        'units_2': 69, 
        'dropout_rate': 0.1938213639314652,
        'activation': 'relu'
    }
    
    # Performance obtenue
    best_value = 0.0042  # MSE sur validation
    # RMSE équivalent: ~291.19 MW

Analyse des résultats
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_optimization_results(study):
        """Analyse complète des résultats d'optimisation."""
        
        print("=== Résumé de l'optimisation ===")
        print(f"Nombre d'essais: {len(study.trials)}")
        print(f"Nombre d'essais complétés: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"Nombre d'essais prunés: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        
        print(f"\n=== Meilleur essai ===")
        print(f"Valeur: {study.best_value:.6f}")
        print(f"Paramètres:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Importance des hyperparamètres
        importance = optuna.importance.get_param_importances(study)
        print(f"\n=== Importance des hyperparamètres ===")
        for param, imp in importance.items():
            print(f"  {param}: {imp:.4f}")

Visualisations des résultats
----------------------------

Graphiques d'optimisation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import optuna.visualization as vis
    import plotly.io as pio
    
    def create_optimization_plots(study):
        """Crée des visualisations de l'optimisation."""
        
        # 1. Historique d'optimisation
        fig1 = vis.plot_optimization_history(study)
        fig1.show()
        
        # 2. Importance des hyperparamètres
        fig2 = vis.plot_param_importances(study)
        fig2.show()
        
        # 3. Coordonnées parallèles
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.show()
        
        # 4. Slice plot
        fig4 = vis.plot_slice(study)
        fig4.show()
        
        # 5. Contour plot (pour paramètres numériques)
        fig5 = vis.plot_contour(study, params=['units_1', 'units_2'])
        fig5.show()

Analyse de convergence
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def plot_convergence_analysis(study):
        """Analyse la convergence de l'optimisation."""
        
        import matplotlib.pyplot as plt
        
        # Extraire les valeurs des essais
        trials = study.trials
        values = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Meilleure valeur cumulée
        best_values = []
        current_best = float('inf')
        
        for value in values:
            if value < current_best:
                current_best = value
            best_values.append(current_best)
        
        # Graphique de convergence
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(values) + 1), values, 'b-', alpha=0.6, label='Valeurs des essais')
        plt.plot(range(1, len(best_values) + 1), best_values, 'r-', linewidth=2, label='Meilleure valeur')
        plt.xlabel('Numéro d\'essai')
        plt.ylabel('Validation Loss')
        plt.title('Convergence de l\'optimisation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Distribution des valeurs
        plt.subplot(1, 2, 2)
        plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(study.best_value, color='red', linestyle='--', 
                   label=f'Meilleure valeur: {study.best_value:.6f}')
        plt.xlabel('Validation Loss')
        plt.ylabel('Fréquence')
        plt.title('Distribution des performances')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

Optimisation multi-objectifs
----------------------------

Fonction objective multi-objectifs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def multi_objective_function(trial):
        """
        Optimisation multi-objectifs : minimiser la loss ET le temps d'entraînement.
        """
        import time
        
        # Paramètres suggérés
        params = {
            'units_1': trial.suggest_int('units_1', 50, 150),
            'units_2': trial.suggest_int('units_2', 30, 100),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5)
        }
        
        model = build_model(**params)
        
        # Mesure du temps d'entraînement
        start_time = time.time()
        
        history = model.fit(
            train_generator,
            epochs=30,
            validation_data=val_generator,
            verbose=0
        )
        
        training_time = time.time() - start_time
        val_loss = min(history.history['val_loss'])
        
        return val_loss, training_time  # Deux objectifs à minimiser

Configuration multi-objectifs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Étude multi-objectifs
    study = optuna.create_study(
        directions=['minimize', 'minimize'],  # Minimiser loss et temps
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )
    
    study.optimize(multi_objective_function, n_trials=50)
    
    # Analyse du front de Pareto
    pareto_solutions = study.best_trials
    
    for i, trial in enumerate(pareto_solutions):
        print(f"Solution Pareto {i+1}:")
        print(f"  Loss: {trial.values[0]:.6f}")
        print(f"  Temps: {trial.values[1]:.2f}s")
        print(f"  Paramètres: {trial.params}")

Stratégies d'optimisation avancées
----------------------------------

Recherche par grille hybride
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def hybrid_optimization():
        """
        Combine recherche par grille et optimisation bayésienne.
        """
        
        # 1. Recherche grossière par grille
        grid_params = {
            'units_1': [50, 75, 100, 125, 150],
            'activation': ['tanh', 'relu']
        }
        
        best_grid_score = float('inf')
        best_grid_params = {}
        
        for units in grid_params['units_1']:
            for activation in grid_params['activation']:
                # Test rapide avec paramètres fixes
                model = build_model(units_1=units, activation=activation)
                score = quick_evaluation(model)  # Évaluation rapide
                
                if score < best_grid_score:
                    best_grid_score = score
                    best_grid_params = {'units_1': units, 'activation': activation}
        
        # 2. Optimisation fine autour des meilleurs paramètres
        def refined_objective(trial):
            # Recherche autour des meilleurs paramètres trouvés
            units_1 = trial.suggest_int('units_1', 
                                       best_grid_params['units_1'] - 20,
                                       best_grid_params['units_1'] + 20)
            # ... autres paramètres
            
            return full_evaluation(build_model(units_1=units_1, ...))
        
        # Optimisation Optuna raffinée
        refined_study = optuna.create_study(direction='minimize')
        refined_study.optimize(refined_objective, n_trials=30)
        
        return refined_study

Optimisation adaptative
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def adaptive_optimization(max_trials=100):
        """
        Optimisation adaptative qui ajuste la stratégie en cours d'exécution.
        """
        
        study = optuna.create_study(direction='minimize')
        
        for trial_number in range(max_trials):
            
            # Adapter la stratégie selon le progrès
            if trial_number < 20:
                # Phase d'exploration
                sampler = optuna.samplers.RandomSampler(seed=42)
            elif trial_number < 60:
                # Phase d'exploitation
                sampler = optuna.samplers.TPESampler(seed=42)
            else:
                # Phase de raffinement
                sampler = optuna.samplers.CmaEsSampler(seed=42)
            
            # Mise à jour du sampler
            study.sampler = sampler
            
            # Un seul essai avec le sampler adapté
            study.optimize(objective_with_pruning, n_trials=1)
            
            # Analyse du progrès
            if trial_number % 10 == 0:
                print(f"Trial {trial_number}: Meilleure valeur = {study.best_value:.6f}")
        
        return study

Bonnes pratiques d'optimisation
-------------------------------

Gestion des ressources
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configuration pour éviter l'épuisement des ressources
    def resource_aware_objective(trial):
        """Fonction objective consciente des ressources."""
        
        # Limiter la complexité du modèle selon les ressources
        max_units = 200 if torch.cuda.is_available() else 100
        
        units_1 = trial.suggest_int('units_1', 30, max_units)
        
        # Surveillance mémoire
        import psutil
        if psutil.virtual_memory().percent > 90:
            raise optuna.TrialPruned("Mémoire insuffisante")
        
        # ... reste de la fonction

Reproductibilité
~~~~~~~~~~~~~~~

.. code-block:: python

    # Assurer la reproductibilité
    def set_seeds(seed=42):
        """Configure tous les seeds pour la reproductibilité."""
        
        import random
        import numpy as np
        import tensorflow as tf
        
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Configuration TensorFlow déterministe
        tf.config.experimental.enable_op_determinism()

Sauvegarde et reprise
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Sauvegarde périodique des résultats
    def optimization_with_checkpoints():
        """Optimisation avec sauvegarde automatique."""
        
        study = optuna.create_study(
            direction='minimize',
            storage='sqlite:///optimization_study.db',
            study_name='lstm_energy_prediction',
            load_if_exists=True  # Reprendre si interrompu
        )
        
        # Callback de sauvegarde
        def checkpoint_callback(study, trial):
            if trial.number % 5 == 0:
                # Sauvegarder les résultats intermédiaires
                results = {
                    'best_params': study.best_params,
                    'best_value': study.best_value,
                    'trial_number': trial.number
                }
                
                import json
                with open(f'checkpoint_{trial.number}.json', 'w') as f:
                    json.dump(results, f)
        
        study.optimize(objective, n_trials=100, callbacks=[checkpoint_callback])
        
        return study

Interprétation des résultats
----------------------------

Analyse de sensibilité
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def sensitivity_analysis(study):
        """Analyse de sensibilité des hyperparamètres."""
        
        # Calculer l'importance relative
        importance = optuna.importance.get_param_importances(study)
        
        # Analyser l'impact de chaque paramètre
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{param}: {imp:.3f} ({imp/sum(importance.values())*100:.1f}%)")
        
        # Recommandations basées sur l'importance
        if importance.get('units_1', 0) > 0.3:
            print("Recommandation: Focalisez l'optimisation sur units_1")
        
        if importance.get('dropout_rate', 0) < 0.1:
            print("Recommandation: dropout_rate peut être fixé")

Validation finale
~~~~~~~~~~~~~~~~

.. code-block:: python

    def final_validation(best_params):
        """Validation finale du modèle avec les meilleurs paramètres."""
        
        # Construire le modèle final
        final_model = build_model(**best_params)
        
        # Entraînement complet
        history = final_model.fit(
            full_train_generator,
            epochs=100,
            validation_data=test_generator,
            callbacks=[EarlyStopping(patience=15)]
        )
        
        # Évaluation sur test set
        test_loss = final_model.evaluate(test_generator)
        
        print(f"Performance finale sur test: {test_loss:.6f}")
        
        return final_model, history

Prochaines étapes
----------------

Avec l'optimisation maîtrisée :

1. Explorez :doc:`model_evaluation` pour évaluer en profondeur
2. Consultez :doc:`notebooks/lstm_training` pour des exemples pratiques
3. Découvrez :doc:`interface` pour déployer vos modèles optimisés
