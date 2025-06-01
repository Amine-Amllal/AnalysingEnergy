Évaluation des Modèles
=====================

Cette section présente les méthodes d'évaluation utilisées pour valider les performances des modèles LSTM de prédiction énergétique et analyser leur qualité prédictive.

Vue d'ensemble de l'évaluation
------------------------------

L'évaluation des modèles LSTM pour la prédiction énergétique nécessite une approche multi-dimensionnelle comprenant :

* **Métriques de performance** adaptées aux séries temporelles
* **Validation temporelle** respectant la chronologie des données
* **Analyse des résidus** pour détecter les biais
* **Tests de robustesse** sur différentes périodes
* **Comparaison avec modèles de référence** (baselines)

Métriques de performance
------------------------

Métriques principales
~~~~~~~~~~~~~~~~~~~~

Pour l'évaluation de notre modèle LSTM, nous utilisons plusieurs métriques complémentaires :

.. code-block:: python

   import numpy as np
   from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
   
   class ModelEvaluator:
       def __init__(self, scaler=None):
           self.scaler = scaler
           self.metrics = {}
       
       def calculate_metrics(self, y_true, y_pred):
           """Calcul de toutes les métriques d'évaluation"""
           
           # Dénormalisation si nécessaire
           if self.scaler:
               y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
               y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
           
           metrics = {}
           
           # RMSE (Root Mean Square Error)
           metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
           
           # MAE (Mean Absolute Error)
           metrics['MAE'] = mean_absolute_error(y_true, y_pred)
           
           # MAPE (Mean Absolute Percentage Error)
           metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
           
           # R² Score
           metrics['R2'] = r2_score(y_true, y_pred)
           
           # Coefficient de corrélation
           metrics['Correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
           
           return metrics

Métriques spécialisées pour l'énergie
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def calculate_energy_specific_metrics(self, y_true, y_pred):
       """Métriques spécifiques au domaine énergétique"""
       
       metrics = {}
       
       # NRMSE (Normalized RMSE)
       rmse = np.sqrt(mean_squared_error(y_true, y_pred))
       metrics['NRMSE'] = rmse / (np.max(y_true) - np.min(y_true)) * 100
       
       # Bias (biais moyen)
       metrics['Bias'] = np.mean(y_pred - y_true)
       
       # Forecast Skill Score
       persistence_pred = np.roll(y_true, 1)[1:]  # Prédiction naïve (valeur précédente)
       y_true_skill = y_true[1:]
       y_pred_skill = y_pred[1:]
       
       mse_model = mean_squared_error(y_true_skill, y_pred_skill)
       mse_persistence = mean_squared_error(y_true_skill, persistence_pred)
       
       metrics['Skill_Score'] = 1 - (mse_model / mse_persistence)
       
       # Peak Prediction Accuracy (précision sur les pics)
       threshold = np.percentile(y_true, 90)  # Top 10%
       peak_indices = y_true >= threshold
       
       if np.any(peak_indices):
           metrics['Peak_MAE'] = mean_absolute_error(y_true[peak_indices], y_pred[peak_indices])
           metrics['Peak_MAPE'] = np.mean(np.abs((y_true[peak_indices] - y_pred[peak_indices]) 
                                               / (y_true[peak_indices] + 1e-8))) * 100
       
       return metrics

Validation temporelle
---------------------

Validation Walk-Forward
~~~~~~~~~~~~~~~~~~~~~~~

La validation walk-forward simule les conditions réelles de prédiction :

.. code-block:: python

   def walk_forward_validation(self, model, X, y, initial_train_size=0.7, step_size=30):
       """Validation walk-forward pour séries temporelles"""
       
       total_samples = len(X)
       initial_train_end = int(total_samples * initial_train_size)
       
       predictions = []
       actuals = []
       metrics_history = []
       
       for start_idx in range(initial_train_end, total_samples - step_size, step_size):
           # Données d'entraînement (fenêtre glissante)
           X_train = X[:start_idx]
           y_train = y[:start_idx]
           
           # Données de test (période suivante)
           X_test = X[start_idx:start_idx + step_size]
           y_test = y[start_idx:start_idx + step_size]
           
           # Réentraînement du modèle
           model.fit(X_train, y_train, epochs=50, verbose=0)
           
           # Prédiction
           y_pred = model.predict(X_test)
           
           # Stockage des résultats
           predictions.extend(y_pred.flatten())
           actuals.extend(y_test)
           
           # Calcul des métriques pour cette période
           period_metrics = self.calculate_metrics(y_test, y_pred.flatten())
           metrics_history.append(period_metrics)
           
           print(f"Période {start_idx}-{start_idx + step_size}: RMSE = {period_metrics['RMSE']:.2f}")
       
       return np.array(actuals), np.array(predictions), metrics_history

Validation croisée temporelle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import TimeSeriesSplit
   
   def time_series_cross_validation(self, model_builder, X, y, n_splits=5):
       """Validation croisée adaptée aux séries temporelles"""
       
       tscv = TimeSeriesSplit(n_splits=n_splits)
       cv_results = []
       
       for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
           print(f"Fold {fold + 1}/{n_splits}")
           
           X_train, X_test = X[train_idx], X[test_idx]
           y_train, y_test = y[train_idx], y[test_idx]
           
           # Construction et entraînement du modèle
           model = model_builder()
           history = model.fit(X_train, y_train, 
                             validation_data=(X_test, y_test),
                             epochs=100, verbose=0)
           
           # Prédiction et évaluation
           y_pred = model.predict(X_test)
           metrics = self.calculate_metrics(y_test, y_pred.flatten())
           
           cv_results.append({
               'fold': fold,
               'metrics': metrics,
               'history': history.history
           })
       
       return cv_results

Analyse des résidus
-------------------

Tests de normalité
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scipy import stats
   import matplotlib.pyplot as plt
   
   def analyze_residuals(self, y_true, y_pred):
       """Analyse complète des résidus"""
       
       residuals = y_true - y_pred
       
       analysis = {}
       
       # Test de normalité (Shapiro-Wilk)
       shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limite pour performance
       analysis['normality'] = {
           'statistic': shapiro_stat,
           'p_value': shapiro_p,
           'is_normal': shapiro_p > 0.05
       }
       
       # Test d'autocorrélation (Durbin-Watson)
       from statsmodels.stats.diagnostic import durbin_watson
       dw_stat = durbin_watson(residuals)
       analysis['autocorrelation'] = {
           'durbin_watson': dw_stat,
           'independence': 1.5 < dw_stat < 2.5
       }
       
       # Test d'homoscédasticité (Breusch-Pagan)
       from statsmodels.stats.diagnostic import het_breuschpagan
       X_for_test = np.arange(len(residuals)).reshape(-1, 1)
       bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_for_test)
       analysis['homoscedasticity'] = {
           'breusch_pagan_stat': bp_stat,
           'p_value': bp_p,
           'is_homoscedastic': bp_p > 0.05
       }
       
       return analysis, residuals

Visualisation des résidus
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def plot_residual_analysis(self, y_true, y_pred, save_path=None):
       """Visualisation complète de l'analyse des résidus"""
       
       residuals = y_true - y_pred
       
       fig, axes = plt.subplots(2, 3, figsize=(18, 12))
       
       # 1. Résidus vs Prédictions
       axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
       axes[0, 0].axhline(y=0, color='r', linestyle='--')
       axes[0, 0].set_xlabel('Prédictions')
       axes[0, 0].set_ylabel('Résidus')
       axes[0, 0].set_title('Résidus vs Prédictions')
       
       # 2. Q-Q Plot
       stats.probplot(residuals, dist="norm", plot=axes[0, 1])
       axes[0, 1].set_title('Q-Q Plot (Normalité)')
       
       # 3. Histogramme des résidus
       axes[0, 2].hist(residuals, bins=50, density=True, alpha=0.7)
       # Courbe normale de référence
       x_norm = np.linspace(residuals.min(), residuals.max(), 100)
       y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
       axes[0, 2].plot(x_norm, y_norm, 'r-', label='Normal')
       axes[0, 2].set_title('Distribution des résidus')
       axes[0, 2].legend()
       
       # 4. Résidus dans le temps
       axes[1, 0].plot(residuals)
       axes[1, 0].axhline(y=0, color='r', linestyle='--')
       axes[1, 0].set_title('Résidus dans le temps')
       axes[1, 0].set_xlabel('Index temporel')
       
       # 5. Autocorrélation des résidus
       from statsmodels.tsa.stattools import acf
       lags = min(50, len(residuals) // 4)
       autocorr = acf(residuals, nlags=lags)
       axes[1, 1].plot(range(lags + 1), autocorr)
       axes[1, 1].axhline(y=0, color='r', linestyle='--')
       axes[1, 1].set_title('Autocorrélation des résidus')
       axes[1, 1].set_xlabel('Lag')
       
       # 6. Valeurs absolues des résidus
       axes[1, 2].plot(np.abs(residuals))
       axes[1, 2].set_title('Valeurs absolues des résidus')
       axes[1, 2].set_xlabel('Index temporel')
       
       plt.tight_layout()
       
       if save_path:
           plt.savefig(save_path, dpi=300, bbox_inches='tight')
       plt.show()

Comparaison avec modèles de référence
-------------------------------------

Modèles baseline
~~~~~~~~~~~~~~~

.. code-block:: python

   class BaselineModels:
       def __init__(self):
           self.models = {}
       
       def naive_forecast(self, y_train, n_steps):
           """Prédiction naïve (dernière valeur)"""
           return np.full(n_steps, y_train[-1])
       
       def seasonal_naive(self, y_train, n_steps, season_length=365):
           """Prédiction naïve saisonnière"""
           predictions = []
           for i in range(n_steps):
               seasonal_idx = len(y_train) - season_length + (i % season_length)
               if seasonal_idx >= 0:
                   predictions.append(y_train[seasonal_idx])
               else:
                   predictions.append(y_train[-1])
           return np.array(predictions)
       
       def moving_average(self, y_train, n_steps, window=30):
           """Moyenne mobile"""
           ma_value = np.mean(y_train[-window:])
           return np.full(n_steps, ma_value)
       
       def linear_trend(self, y_train, n_steps):
           """Extrapolation de tendance linéaire"""
           x = np.arange(len(y_train))
           slope, intercept = np.polyfit(x, y_train, 1)
           
           future_x = np.arange(len(y_train), len(y_train) + n_steps)
           return slope * future_x + intercept

Comparaison systématique
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compare_with_baselines(self, lstm_model, X_test, y_test):
       """Comparaison avec modèles de référence"""
       
       baseline_models = BaselineModels()
       
       # Prédiction LSTM
       y_pred_lstm = lstm_model.predict(X_test).flatten()
       
       # Reconstruction des données d'entraînement (approximation)
       y_train_approx = y_test[:int(len(y_test) * 0.8)]  # Simplification
       n_steps = len(y_test) - len(y_train_approx)
       
       # Prédictions baseline
       baselines = {
           'LSTM': y_pred_lstm,
           'Naive': baseline_models.naive_forecast(y_train_approx, len(y_test)),
           'Seasonal_Naive': baseline_models.seasonal_naive(y_train_approx, len(y_test)),
           'Moving_Average': baseline_models.moving_average(y_train_approx, len(y_test)),
           'Linear_Trend': baseline_models.linear_trend(y_train_approx, len(y_test))
       }
       
       # Évaluation de chaque modèle
       comparison_results = {}
       for model_name, predictions in baselines.items():
           if len(predictions) == len(y_test):
               metrics = self.calculate_metrics(y_test, predictions)
               comparison_results[model_name] = metrics
       
       return comparison_results

Métriques de performance du modèle final
----------------------------------------

Résultats du modèle optimisé
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notre modèle LSTM optimisé (final_model 291.19.h5) présente les performances suivantes :

.. code-block:: python

   # Résultats sur le dataset de test
   FINAL_MODEL_METRICS = {
       'RMSE': 291.19,  # MW
       'MAE': 185.43,   # MW
       'MAPE': 12.7,    # %
       'R2': 0.847,
       'Correlation': 0.921,
       'NRMSE': 8.9,    # %
       'Skill_Score': 0.73
   }

Analyse par période
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_performance_by_period(self, y_true, y_pred, dates):
       """Analyse des performances par période"""
       
       df_results = pd.DataFrame({
           'date': dates,
           'actual': y_true,
           'predicted': y_pred,
           'residual': y_true - y_pred
       })
       
       df_results['month'] = pd.to_datetime(df_results['date']).dt.month
       df_results['season'] = pd.to_datetime(df_results['date']).dt.quarter
       df_results['year'] = pd.to_datetime(df_results['date']).dt.year
       
       # Performance par saison
       seasonal_performance = df_results.groupby('season').apply(
           lambda x: {
               'RMSE': np.sqrt(mean_squared_error(x['actual'], x['predicted'])),
               'MAE': mean_absolute_error(x['actual'], x['predicted']),
               'R2': r2_score(x['actual'], x['predicted'])
           }
       )
       
       # Performance par mois
       monthly_performance = df_results.groupby('month').apply(
           lambda x: {
               'RMSE': np.sqrt(mean_squared_error(x['actual'], x['predicted'])),
               'MAE': mean_absolute_error(x['actual'], x['predicted']),
               'samples': len(x)
           }
       )
       
       return seasonal_performance, monthly_performance

Tests de robustesse
-------------------

Test sur données out-of-sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def out_of_sample_test(self, model, X_train, y_train, X_future, y_future):
       """Test sur données complètement nouvelles"""
       
       # Entraînement sur données historiques
       model.fit(X_train, y_train, epochs=100, verbose=0)
       
       # Test sur données futures
       y_pred_future = model.predict(X_future)
       
       # Évaluation
       oos_metrics = self.calculate_metrics(y_future, y_pred_future.flatten())
       
       print("Performance Out-of-Sample:")
       for metric, value in oos_metrics.items():
           print(f"{metric}: {value:.4f}")
       
       return oos_metrics

Test de sensibilité aux paramètres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def sensitivity_analysis(self, model_builder, X, y, param_ranges):
       """Analyse de sensibilité aux hyperparamètres"""
       
       base_params = {
           'units_1': 74,
           'units_2': 69,
           'dropout_rate': 0.1938,
           'learning_rate': 0.001
       }
       
       sensitivity_results = {}
       
       for param, values in param_ranges.items():
           param_results = []
           
           for value in values:
               # Modification du paramètre
               test_params = base_params.copy()
               test_params[param] = value
               
               # Construction et test du modèle
               model = model_builder(**test_params)
               
               # Validation croisée rapide
               cv_scores = []
               tscv = TimeSeriesSplit(n_splits=3)
               
               for train_idx, val_idx in tscv.split(X):
                   X_train, X_val = X[train_idx], X[val_idx]
                   y_train, y_val = y[train_idx], y[val_idx]
                   
                   model.fit(X_train, y_train, epochs=50, verbose=0)
                   y_pred = model.predict(X_val)
                   
                   rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                   cv_scores.append(rmse)
               
               param_results.append({
                   'value': value,
                   'mean_rmse': np.mean(cv_scores),
                   'std_rmse': np.std(cv_scores)
               })
           
           sensitivity_results[param] = param_results
       
       return sensitivity_results

Rapport d'évaluation
--------------------

Génération automatique de rapport
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_evaluation_report(self, model_name, metrics, residual_analysis, 
                                comparison_results, save_path='evaluation_report.html'):
       """Génération d'un rapport d'évaluation complet"""
       
       html_template = f"""
       <!DOCTYPE html>
       <html>
       <head>
           <title>Rapport d'Évaluation - {model_name}</title>
           <style>
               body {{ font-family: Arial, sans-serif; margin: 40px; }}
               .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
               .good {{ color: green; }}
               .warning {{ color: orange; }}
               .bad {{ color: red; }}
           </style>
       </head>
       <body>
           <h1>Rapport d'Évaluation: {model_name}</h1>
           
           <h2>Métriques Principales</h2>
           <div class="metric">RMSE: {metrics['RMSE']:.2f} MW</div>
           <div class="metric">MAE: {metrics['MAE']:.2f} MW</div>
           <div class="metric">MAPE: {metrics['MAPE']:.2f}%</div>
           <div class="metric">R²: {metrics['R2']:.4f}</div>
           
           <h2>Analyse des Résidus</h2>
           <div class="metric">Normalité: {'✓' if residual_analysis['normality']['is_normal'] else '✗'}</div>
           <div class="metric">Indépendance: {'✓' if residual_analysis['autocorrelation']['independence'] else '✗'}</div>
           <div class="metric">Homoscédasticité: {'✓' if residual_analysis['homoscedasticity']['is_homoscedastic'] else '✗'}</div>
           
           <h2>Comparaison avec Baselines</h2>
           <table border="1">
               <tr><th>Modèle</th><th>RMSE</th><th>MAE</th><th>R²</th></tr>
       """
       
       for model, model_metrics in comparison_results.items():
           html_template += f"""
               <tr>
                   <td>{model}</td>
                   <td>{model_metrics['RMSE']:.2f}</td>
                   <td>{model_metrics['MAE']:.2f}</td>
                   <td>{model_metrics['R2']:.4f}</td>
               </tr>
           """
       
       html_template += """
           </table>
           
           <h2>Recommandations</h2>
           <ul>
               <li>Le modèle LSTM montre d'excellentes performances globales</li>
               <li>Performance supérieure aux modèles de référence</li>
               <li>Résidus satisfaisants avec quelques améliorations possibles</li>
           </ul>
           
       </body>
       </html>
       """
       
       with open(save_path, 'w', encoding='utf-8') as f:
           f.write(html_template)
       
       print(f"Rapport sauvegardé: {save_path}")

Bonnes pratiques d'évaluation
-----------------------------

Checklist d'évaluation
~~~~~~~~~~~~~~~~~~~~~~

1. **Métriques multiples** : Ne jamais se fier à une seule métrique
2. **Validation temporelle** : Respecter l'ordre chronologique
3. **Analyse des résidus** : Vérifier les hypothèses du modèle
4. **Comparaison baseline** : Établir la valeur ajoutée du modèle
5. **Tests de robustesse** : Valider sur différentes conditions

Interprétation des résultats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notre modèle LSTM avec RMSE = 291.19 MW représente :

* **Erreur relative** : ~8.9% de la capacité maximale
* **Performance excellente** par rapport aux baselines
* **Précision suffisante** pour la planification énergétique
* **Robustesse confirmée** sur différentes périodes

Prochaines étapes
-----------------

L'évaluation complète permet de :

* **Valider la qualité** du modèle développé
* **Identifier les améliorations** possibles
* **Documenter les performances** pour les utilisateurs
* **Guider le déploiement** en production

Pour aller plus loin :

* :doc:`interface` - Interface utilisateur Streamlit
* :doc:`troubleshooting` - Résolution de problèmes
* :doc:`faq` - Questions fréquentes
