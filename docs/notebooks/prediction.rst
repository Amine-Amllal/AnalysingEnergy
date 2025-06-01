Notebook de prédiction
====================

Ce notebook présente l'utilisation des modèles LSTM entraînés pour faire des prédictions de production d'énergie verte.

Vue d'ensemble
--------------

Le notebook ``Predicting_next_365days.ipynb`` utilise les modèles LSTM optimisés pour générer des prédictions sur différents horizons temporels, de quelques jours à une année complète.

Objectifs du notebook
---------------------

1. **Chargement des modèles pré-entraînés**
2. **Préparation des données d'entrée**
3. **Génération de prédictions multi-horizons**
4. **Analyse de l'incertitude des prédictions**
5. **Visualisation des résultats**
6. **Export et sauvegarde des prédictions**

Structure du notebook
---------------------

Imports et configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timedelta
    
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import joblib
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import warnings
    warnings.filterwarnings('ignore')
    
    # Configuration des graphiques
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

Chargement des modèles et scalers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def load_trained_models():
        """Charge les modèles et scalers pré-entraînés."""
        
        try:
            # Chargement du modèle principal
            model = load_model('Notebooks/models/final_model 291.19.h5')
            print("✅ Modèle principal chargé avec succès")
            
            # Chargement des scalers
            scaler_X = joblib.load('Notebooks/scalers/X_train_scaler.pkl')
            scaler_y = joblib.load('Notebooks/scalers/y_train_scaler.pkl')
            print("✅ Scalers chargés avec succès")
            
            # Chargement des modèles spécialisés (optionnel)
            specialized_models = {}
            model_files = [
                'temp2_max(c)_LSTM.h5', 'temp2_min(c)_LSTM.h5',
                'wind_speed50_ave(ms)_LSTM.h5', 'suface_pressure(pa)_LSTM.h5'
            ]
            
            for model_file in model_files:
                try:
                    model_path = f'Notebooks/models/{model_file}'
                    specialized_models[model_file] = load_model(model_path)
                    print(f"✅ Modèle spécialisé {model_file} chargé")
                except:
                    print(f"⚠️ Modèle {model_file} non trouvé")
            
            return model, scaler_X, scaler_y, specialized_models
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return None, None, None, {}

Préparation des données d'entrée
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Génération de données météorologiques synthétiques :**

.. code-block:: python

    def generate_weather_forecast(start_date, days_ahead=365):
        """Génère des prévisions météorologiques synthétiques."""
        
        dates = pd.date_range(start=start_date, periods=days_ahead, freq='D')
        
        # Paramètres saisonniers réalistes
        day_of_year = dates.dayofyear
        
        # Température (avec cycle saisonnier)
        temp_base = 20 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temp_max = temp_base + 5 + np.random.normal(0, 2, len(dates))
        temp_min = temp_base - 5 + np.random.normal(0, 2, len(dates))
        temp_ave = (temp_max + temp_min) / 2
        
        # Pression atmosphérique
        pressure_base = 101300  # Pa
        pressure = pressure_base + np.random.normal(0, 500, len(dates))
        
        # Vitesse du vent (avec variation saisonnière)
        wind_base = 5 + 3 * np.sin(2 * np.pi * (day_of_year - 365/4) / 365)
        wind_max = wind_base + np.random.exponential(2, len(dates))
        wind_min = np.maximum(0, wind_base - np.random.exponential(1, len(dates)))
        wind_ave = (wind_max + wind_min) / 2
        
        # Précipitations
        precip = np.random.exponential(0.5, len(dates))
        
        # Demande d'énergie (avec patterns réalistes)
        demand_base = 7000 + 1000 * np.sin(2 * np.pi * (day_of_year - 365/6) / 365)
        demand = demand_base + np.random.normal(0, 300, len(dates))
        
        forecast_data = pd.DataFrame({
            'date': dates,
            'temp2_max(c)': temp_max,
            'temp2_min(c)': temp_min,
            'temp2_ave(c)': temp_ave,
            'suface_pressure(pa)': pressure,
            'wind_speed50_max(ms)': wind_max,
            'wind_speed50_min(ms)': wind_min,
            'wind_speed50_ave(ms)': wind_ave,
            'prectotcorr': precip,
            'total_demand(mw)': demand
        })
        
        return forecast_data

**Chargement de données météorologiques réelles :**

.. code-block:: python

    def load_real_weather_data(file_path='Data/weather_forecast.csv'):
        """Charge des données météorologiques réelles si disponibles."""
        
        try:
            weather_data = pd.read_csv(file_path, parse_dates=['date'])
            print(f"✅ Données météo réelles chargées: {len(weather_data)} jours")
            return weather_data
        except FileNotFoundError:
            print("⚠️ Fichier de prévisions météo non trouvé, utilisation de données synthétiques")
            return None

Génération de prédictions
~~~~~~~~~~~~~~~~~~~~~~~~~

**Prédiction simple (jour suivant) :**

.. code-block:: python

    def predict_next_day(model, scaler_X, scaler_y, last_sequence):
        """Prédit la production du jour suivant."""
        
        # Préparation de l'entrée
        input_scaled = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        
        # Prédiction
        prediction_scaled = model.predict(input_scaled, verbose=0)
        
        # Dénormalisation
        prediction = scaler_y.inverse_transform(prediction_scaled)
        
        return prediction[0, 0]

**Prédiction multi-étapes (approche récursive) :**

.. code-block:: python

    def predict_recursive(model, scaler_X, scaler_y, initial_sequence, 
                         future_weather, days_ahead=30):
        """Prédiction récursive pour plusieurs jours."""
        
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for day in range(days_ahead):
            # Prédiction du jour actuel
            pred = predict_next_day(model, scaler_X, scaler_y, current_sequence)
            predictions.append(pred)
            
            # Mise à jour de la séquence pour le jour suivant
            if day < len(future_weather) - 1:
                # Nouvelles données météo pour le jour suivant
                new_features = future_weather.iloc[day].values
                new_features_scaled = scaler_X.transform([new_features])
                
                # Déplacement de la fenêtre temporelle
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = new_features_scaled[0]
        
        return np.array(predictions)

**Prédiction avec ensemble de modèles :**

.. code-block:: python

    def ensemble_prediction(models, scaler_X, scaler_y, input_data, weights=None):
        """Prédiction en utilisant un ensemble de modèles."""
        
        if weights is None:
            weights = [1.0] * len(models)
        
        predictions = []
        
        for model in models:
            pred_scaled = model.predict(input_data, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)
            predictions.append(pred.flatten())
        
        # Moyenne pondérée
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        # Calcul de l'incertitude (écart-type des prédictions)
        uncertainty = np.std(predictions, axis=0)
        
        return weighted_pred, uncertainty

Prédictions à long terme (365 jours)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Approche segmentée :**

.. code-block:: python

    def predict_long_term(model, scaler_X, scaler_y, initial_data, 
                         weather_forecast, segment_length=30):
        """Prédiction à long terme par segments pour réduire l'erreur cumulative."""
        
        total_days = len(weather_forecast)
        all_predictions = []
        
        # Division en segments
        for start_day in range(0, total_days, segment_length):
            end_day = min(start_day + segment_length, total_days)
            segment_weather = weather_forecast.iloc[start_day:end_day]
            
            if start_day == 0:
                # Premier segment : utiliser les données initiales
                segment_predictions = predict_recursive(
                    model, scaler_X, scaler_y, initial_data, 
                    segment_weather, len(segment_weather)
                )
            else:
                # Segments suivants : réinitialiser avec les données récentes
                # (approche simplifiée, pourrait être améliorée)
                segment_predictions = predict_recursive(
                    model, scaler_X, scaler_y, initial_data, 
                    segment_weather, len(segment_weather)
                )
            
            all_predictions.extend(segment_predictions)
            
            print(f"Segment {start_day//segment_length + 1}: "
                  f"jours {start_day+1}-{end_day} traités")
        
        return np.array(all_predictions)

**Prédiction avec réajustement adaptatif :**

.. code-block:: python

    def adaptive_prediction(model, scaler_X, scaler_y, historical_data, 
                           weather_forecast, recalibration_interval=7):
        """Prédiction avec recalibration périodique."""
        
        predictions = []
        current_sequence = historical_data[-60:]  # Dernières 60 observations
        
        for day in range(len(weather_forecast)):
            # Prédiction du jour courant
            pred = predict_next_day(model, scaler_X, scaler_y, current_sequence)
            predictions.append(pred)
            
            # Mise à jour de la séquence
            new_features = weather_forecast.iloc[day].values
            new_features_scaled = scaler_X.transform([new_features])
            
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_features_scaled[0]
            
            # Recalibration périodique (si données réelles disponibles)
            if (day + 1) % recalibration_interval == 0:
                print(f"Recalibration au jour {day + 1}")
                # Ici, on pourrait réentraîner ou ajuster le modèle
        
        return np.array(predictions)

Analyse d'incertitude
~~~~~~~~~~~~~~~~~~~~

**Prédictions avec intervalles de confiance :**

.. code-block:: python

    def prediction_with_uncertainty(model, scaler_X, scaler_y, input_data, 
                                   n_samples=100):
        """Génère des prédictions avec intervalles de confiance Monte Carlo."""
        
        # Activation du dropout pendant l'inférence pour Monte Carlo
        predictions = []
        
        for _ in range(n_samples):
            # Prédiction avec dropout activé
            pred_scaled = model(input_data, training=True)
            pred = scaler_y.inverse_transform(pred_scaled.numpy())
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        # Calcul des statistiques
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Intervalles de confiance
        confidence_95 = {
            'lower': np.percentile(predictions, 2.5, axis=0),
            'upper': np.percentile(predictions, 97.5, axis=0)
        }
        
        confidence_80 = {
            'lower': np.percentile(predictions, 10, axis=0),
            'upper': np.percentile(predictions, 90, axis=0)
        }
        
        return mean_pred, std_pred, confidence_95, confidence_80

**Analyse de sensibilité :**

.. code-block:: python

    def sensitivity_analysis(model, scaler_X, scaler_y, base_input, 
                           feature_names, perturbation_range=0.1):
        """Analyse de sensibilité des prédictions aux variables d'entrée."""
        
        base_pred = predict_next_day(model, scaler_X, scaler_y, base_input)
        sensitivities = {}
        
        for i, feature in enumerate(feature_names):
            # Perturbation positive
            perturbed_input_pos = base_input.copy()
            perturbed_input_pos[:, i] *= (1 + perturbation_range)
            pred_pos = predict_next_day(model, scaler_X, scaler_y, perturbed_input_pos)
            
            # Perturbation négative
            perturbed_input_neg = base_input.copy()
            perturbed_input_neg[:, i] *= (1 - perturbation_range)
            pred_neg = predict_next_day(model, scaler_X, scaler_y, perturbed_input_neg)
            
            # Sensibilité
            sensitivity = (pred_pos - pred_neg) / (2 * perturbation_range * base_pred)
            sensitivities[feature] = sensitivity
        
        return sensitivities

Visualisation des résultats
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Graphique des prédictions à long terme :**

.. code-block:: python

    def plot_long_term_predictions(dates, predictions, historical_data=None, 
                                  confidence_intervals=None):
        """Visualise les prédictions à long terme."""
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Données historiques si disponibles
        if historical_data is not None:
            hist_dates = historical_data['date']
            hist_values = historical_data['max_generation(mw)']
            ax.plot(hist_dates, hist_values, 'b-', alpha=0.7, 
                   label='Données historiques', linewidth=1)
        
        # Prédictions
        ax.plot(dates, predictions, 'r-', alpha=0.8, 
               label='Prédictions', linewidth=2)
        
        # Intervalles de confiance
        if confidence_intervals is not None:
            ax.fill_between(dates, 
                           confidence_intervals['lower'], 
                           confidence_intervals['upper'],
                           alpha=0.2, color='red', 
                           label='Intervalle 95%')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Production d\'énergie (MW)')
        ax.set_title('Prédictions de Production d\'Énergie - 365 jours')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Formatage des dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

**Analyse comparative mensuelle :**

.. code-block:: python

    def plot_monthly_analysis(dates, predictions):
        """Analyse des prédictions par mois."""
        
        df_pred = pd.DataFrame({
            'date': dates,
            'prediction': predictions
        })
        
        df_pred['month'] = df_pred['date'].dt.month
        df_pred['month_name'] = df_pred['date'].dt.strftime('%B')
        
        # Statistiques mensuelles
        monthly_stats = df_pred.groupby(['month', 'month_name'])['prediction'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Production moyenne par mois
        axes[0, 0].bar(monthly_stats['month_name'], monthly_stats['mean'])
        axes[0, 0].set_title('Production Moyenne par Mois')
        axes[0, 0].set_ylabel('Production (MW)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Variabilité par mois
        axes[0, 1].bar(monthly_stats['month_name'], monthly_stats['std'])
        axes[0, 1].set_title('Variabilité par Mois')
        axes[0, 1].set_ylabel('Écart-type (MW)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Box plot mensuel
        monthly_data = [df_pred[df_pred['month'] == m]['prediction'].values 
                       for m in range(1, 13)]
        axes[1, 0].boxplot(monthly_data)
        axes[1, 0].set_title('Distribution Mensuelle')
        axes[1, 0].set_xlabel('Mois')
        axes[1, 0].set_ylabel('Production (MW)')
        
        # Évolution dans l'année
        axes[1, 1].plot(df_pred['date'], df_pred['prediction'])
        axes[1, 1].set_title('Évolution Annuelle')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Production (MW)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return monthly_stats

Export et sauvegarde
~~~~~~~~~~~~~~~~~~~

**Sauvegarde des prédictions :**

.. code-block:: python

    def save_predictions(dates, predictions, confidence_intervals=None, 
                        output_file='predictions_365_days.csv'):
        """Sauvegarde les prédictions dans un fichier CSV."""
        
        df_output = pd.DataFrame({
            'date': dates,
            'predicted_generation_mw': predictions
        })
        
        if confidence_intervals is not None:
            df_output['confidence_lower_95'] = confidence_intervals['lower']
            df_output['confidence_upper_95'] = confidence_intervals['upper']
        
        # Ajout de métadonnées
        df_output['prediction_date'] = datetime.now()
        df_output['model_version'] = 'final_model_291.19'
        
        # Sauvegarde
        df_output.to_csv(f'Data/{output_file}', index=False)
        print(f"✅ Prédictions sauvegardées dans Data/{output_file}")
        
        return df_output

**Export pour visualisation externe :**

.. code-block:: python

    def export_for_dashboard(predictions_df, metadata=None):
        """Prépare les données pour export vers dashboard/interface."""
        
        # Format pour JSON
        export_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'model_version': 'final_model_291.19',
                'forecast_horizon': len(predictions_df),
                'start_date': predictions_df['date'].min().isoformat(),
                'end_date': predictions_df['date'].max().isoformat()
            },
            'predictions': predictions_df.to_dict(orient='records')
        }
        
        # Ajout de métadonnées optionnelles
        if metadata:
            export_data['metadata'].update(metadata)
        
        # Sauvegarde JSON
        import json
        with open('Data/predictions_dashboard.json', 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print("✅ Données exportées pour dashboard")
        
        return export_data

Pipeline de prédiction complet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fonction principale :**

.. code-block:: python

    def run_prediction_pipeline(start_date=None, days_ahead=365, 
                               use_real_weather=False):
        """Pipeline complet de prédiction."""
        
        print("🚀 Début du pipeline de prédiction")
        
        # 1. Chargement des modèles
        model, scaler_X, scaler_y, specialized_models = load_trained_models()
        if model is None:
            print("❌ Impossible de charger les modèles")
            return None
        
        # 2. Préparation des données météo
        if start_date is None:
            start_date = datetime.now().date()
        
        if use_real_weather:
            weather_data = load_real_weather_data()
            if weather_data is None:
                weather_data = generate_weather_forecast(start_date, days_ahead)
        else:
            weather_data = generate_weather_forecast(start_date, days_ahead)
        
        print(f"📊 Données météo préparées pour {len(weather_data)} jours")
        
        # 3. Préparation de la séquence initiale
        # (utiliser les dernières données historiques)
        historical_data = pd.read_csv('Data/data.csv')
        last_60_days = historical_data.tail(60)
        
        feature_columns = ['temp2_max(c)', 'temp2_min(c)', 'temp2_ave(c)',
                          'suface_pressure(pa)', 'wind_speed50_max(ms)', 
                          'wind_speed50_min(ms)', 'wind_speed50_ave(ms)',
                          'prectotcorr', 'total_demand(mw)']
        
        initial_sequence = scaler_X.transform(last_60_days[feature_columns].values)
        
        # 4. Génération des prédictions
        print("🔮 Génération des prédictions...")
        predictions = predict_long_term(
            model, scaler_X, scaler_y, initial_sequence, 
            weather_data[feature_columns], segment_length=30
        )
        
        # 5. Calcul des intervalles de confiance
        print("📈 Calcul des intervalles de confiance...")
        dates = pd.date_range(start=start_date, periods=len(predictions), freq='D')
        
        # 6. Visualisation
        plot_long_term_predictions(dates, predictions)
        monthly_stats = plot_monthly_analysis(dates, predictions)
        
        # 7. Sauvegarde
        results_df = save_predictions(dates, predictions)
        export_data = export_for_dashboard(results_df)
        
        print("✅ Pipeline de prédiction terminé avec succès")
        
        return {
            'predictions': predictions,
            'dates': dates,
            'monthly_stats': monthly_stats,
            'results_df': results_df,
            'export_data': export_data
        }

Utilisation pratique
-------------------

**Exécution basique :**

.. code-block:: python

    # Prédiction pour les 365 prochains jours
    results = run_prediction_pipeline()

**Exécution personnalisée :**

.. code-block:: python

    # Prédiction pour 90 jours à partir d'une date spécifique
    from datetime import date
    results = run_prediction_pipeline(
        start_date=date(2024, 1, 1), 
        days_ahead=90,
        use_real_weather=False
    )

**Analyse de cas spécifiques :**

.. code-block:: python

    # Scénario météo extrême
    extreme_weather = generate_weather_forecast(
        datetime.now().date(), 30
    )
    # Modification pour conditions extrêmes
    extreme_weather['temp2_max(c)'] += 10  # Canicule
    extreme_weather['wind_speed50_ave(ms)'] *= 1.5  # Vent fort
    
    # Prédiction avec ce scénario
    # ... code de prédiction avec extreme_weather

Cas d'usage pratiques
~~~~~~~~~~~~~~~~~~~

**1. Planification énergétique quotidienne :**

.. code-block:: python

    def daily_forecast():
        """Prédiction pour les 7 prochains jours."""
        results = run_prediction_pipeline(days_ahead=7)
        print("📅 Prévisions hebdomadaires générées")
        return results['predictions']

**2. Analyse saisonnière :**

.. code-block:: python

    def seasonal_analysis():
        """Analyse des patterns saisonniers."""
        results = run_prediction_pipeline(days_ahead=365)
        monthly_stats = results['monthly_stats']
        
        # Identification du pic de production
        peak_month = monthly_stats.loc[monthly_stats['mean'].idxmax(), 'month_name']
        print(f"🏔️ Pic de production prévu en {peak_month}")
        
        return monthly_stats

**3. Évaluation de scénarios :**

.. code-block:: python

    def scenario_comparison():
        """Compare différents scénarios météorologiques."""
        scenarios = {
            'Normal': run_prediction_pipeline(days_ahead=30),
            'Chaud': run_prediction_pipeline(days_ahead=30),  # avec T+5°C
            'Venteux': run_prediction_pipeline(days_ahead=30)  # avec vent*1.5
        }
        
        for name, scenario in scenarios.items():
            mean_prod = np.mean(scenario['predictions'])
            print(f"{name}: {mean_prod:.0f} MW moyen")

Prochaines étapes
----------------

Après avoir utilisé ce notebook de prédiction :

1. Intégrez les résultats dans :doc:`../interface` pour la visualisation
2. Consultez :doc:`../model_evaluation` pour valider les prédictions
3. Explorez :doc:`data_preprocessing` pour améliorer les données d'entrée

Troubleshooting
---------------

**Problèmes de prédiction :**

- **Prédictions irréalistes** : Vérifiez la normalisation des données d'entrée
- **Erreur de forme** : Assurez-vous que les dimensions correspondent au modèle
- **Performance lente** : Réduisez la fréquence de prédiction ou utilisez GPU

Pour plus d'aide, consultez :doc:`../troubleshooting`.
