Dépannage et Résolution de Problèmes
====================================

Cette section aide à résoudre les problèmes courants rencontrés lors de l'utilisation du projet AnalysingEnergy.

Problèmes d'installation
------------------------

Erreurs lors de l'installation des dépendances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problème** : Erreurs lors de `pip install -r requirements.txt`

**Solutions** :

1. **Mise à jour de pip**
   
   .. code-block:: bash
   
      python -m pip install --upgrade pip

2. **Installation avec cache désactivé**
   
   .. code-block:: bash
   
      pip install --no-cache-dir -r requirements.txt

3. **Installation par étapes**
   
   .. code-block:: bash
   
      # Installation des dépendances principales d'abord
      pip install pandas numpy matplotlib
      pip install tensorflow scikit-learn
      pip install streamlit plotly optuna

**Problème** : Conflit de versions TensorFlow

.. code-block:: text

   ERROR: tensorflow 2.x.x has requirement numpy>=1.19.2, but you have numpy 1.18.0

**Solution** :

.. code-block:: bash

   pip install --upgrade numpy
   pip install tensorflow==2.13.0  # Version stable recommandée

Problèmes avec CUDA/GPU
~~~~~~~~~~~~~~~~~~~~~~~

**Problème** : TensorFlow ne détecte pas le GPU

**Vérification** :

.. code-block:: python

   import tensorflow as tf
   print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
   print(f"CUDA construit: {tf.test.is_built_with_cuda()}")

**Solutions** :

1. **Installation TensorFlow-GPU**
   
   .. code-block:: bash
   
      pip uninstall tensorflow
      pip install tensorflow-gpu==2.13.0

2. **Vérification des versions CUDA/cuDNN**
   
   - TensorFlow 2.13: CUDA 11.8, cuDNN 8.6
   - Télécharger depuis le site NVIDIA

3. **Variables d'environnement**
   
   .. code-block:: bash
   
      set CUDA_VISIBLE_DEVICES=0
      set TF_FORCE_GPU_ALLOW_GROWTH=true

Problèmes de données
-------------------

Fichier de données non trouvé
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Erreur** :

.. code-block:: text

   FileNotFoundError: [Errno 2] No such file or directory: 'Data/data.csv'

**Solutions** :

1. **Vérification du chemin**
   
   .. code-block:: python
   
      import os
      print(f"Répertoire courant: {os.getcwd()}")
      print(f"Fichiers disponibles: {os.listdir('.')}")

2. **Utilisation de chemins absolus**
   
   .. code-block:: python
   
      import os
      data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'data.csv')
      data = pd.read_csv(data_path)

3. **Téléchargement du dataset**
   
   Si le fichier est manquant, assurez-vous d'avoir le dataset BanE-16 complet.

Format de données incorrect
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Erreur** :

.. code-block:: text

   ValueError: Les colonnes requises sont manquantes: ['max_generation(mw)']

**Solution** :

.. code-block:: python

   # Vérification des colonnes
   required_columns = [
       'min_temperature', 'mean_temperature', 'max_temperature',
       'min_windspeed', 'mean_windspeed', 'max_windspeed',
       'total_precipitation', 'surface_pressure', 'mean_relative_humidity',
       'max_generation(mw)'
   ]
   
   missing_columns = set(required_columns) - set(data.columns)
   if missing_columns:
       print(f"Colonnes manquantes: {missing_columns}")
       # Mapper les noms de colonnes si nécessaire
       column_mapping = {
           'generation_max': 'max_generation(mw)',
           'temp_min': 'min_temperature',
           # ... autres mappings
       }
       data = data.rename(columns=column_mapping)

Valeurs manquantes excessives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problème** : Trop de valeurs manquantes dans le dataset

**Diagnostic** :

.. code-block:: python

   # Analyse des valeurs manquantes
   missing_stats = data.isnull().sum()
   missing_pct = (missing_stats / len(data)) * 100
   
   print("Pourcentage de valeurs manquantes par colonne:")
   for col, pct in missing_pct.items():
       if pct > 0:
           print(f"{col}: {pct:.2f}%")

**Solutions** :

1. **Interpolation pour valeurs manquantes < 10%**
   
   .. code-block:: python
   
      # Interpolation linéaire
      data[column] = data[column].interpolate(method='linear')

2. **Suppression pour valeurs manquantes > 50%**
   
   .. code-block:: python
   
      # Suppression des colonnes avec trop de valeurs manquantes
      threshold = 0.5
      data = data.loc[:, data.isnull().mean() < threshold]

3. **Imputation avancée**
   
   .. code-block:: python
   
      from sklearn.impute import KNNImputer
      
      imputer = KNNImputer(n_neighbors=5)
      numeric_columns = data.select_dtypes(include=[np.number]).columns
      data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

Problèmes de modèles
-------------------

Modèle pré-entraîné non trouvé
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Erreur** :

.. code-block:: text

   FileNotFoundError: No file found at 'Notebooks/models/final_model 291.19.h5'

**Solutions** :

1. **Vérification des fichiers de modèles**
   
   .. code-block:: python
   
      import os
      models_dir = 'Notebooks/models/'
      if os.path.exists(models_dir):
           model_files = os.listdir(models_dir)
           print(f"Modèles disponibles: {model_files}")
       else:
           print("Répertoire de modèles non trouvé")

2. **Entraînement d'un nouveau modèle**
   
   .. code-block:: python
   
      # Si aucun modèle pré-entraîné n'est disponible
      predictor = EnergyPredictor()
      predictor.load_data('Data/data.csv')
      history = predictor.train_models(epochs=100)

3. **Téléchargement des modèles pré-entraînés**
   
   Assurez-vous d'avoir les modèles dans le répertoire Notebooks/models/

Erreurs d'entraînement
~~~~~~~~~~~~~~~~~~~~~~

**Problème** : Le modèle ne converge pas

**Symptômes** :
- Loss qui stagne
- Oscillations importantes
- NaN dans les prédictions

**Solutions** :

1. **Réduction du learning rate**
   
   .. code-block:: python
   
      from tensorflow.keras.optimizers import Adam
      
      optimizer = Adam(learning_rate=0.0001)  # Au lieu de 0.001
      model.compile(optimizer=optimizer, loss='mse')

2. **Normalisation des données**
   
   .. code-block:: python
   
      from sklearn.preprocessing import StandardScaler
      
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)

3. **Clipping du gradient**
   
   .. code-block:: python
   
      optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

4. **Réduction de la complexité du modèle**
   
   .. code-block:: python
   
      # Modèle plus simple
      model = Sequential([
           LSTM(32, return_sequences=False, input_shape=input_shape),  # Au lieu de 74
           Dense(1)
       ])

Problème de mémoire GPU
~~~~~~~~~~~~~~~~~~~~~~

**Erreur** :

.. code-block:: text

   ResourceExhaustedError: OOM when allocating tensor

**Solutions** :

1. **Réduction de la taille des batches**
   
   .. code-block:: python
   
      # Réduire batch_size
      model.fit(X_train, y_train, batch_size=16, epochs=100)  # Au lieu de 32

2. **Croissance mémoire GPU**
   
   .. code-block:: python
   
      import tensorflow as tf
      
      gpus = tf.config.experimental.list_physical_devices('GPU')
      if gpus:
           try:
               for gpu in gpus:
                   tf.config.experimental.set_memory_growth(gpu, True)
           except RuntimeError as e:
               print(e)

3. **Limitation de mémoire GPU**
   
   .. code-block:: python
   
      # Limiter à 4GB par exemple
      tf.config.experimental.set_memory_growth(gpu, True)
      tf.config.experimental.set_virtual_device_configuration(
           gpu,
           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
       )

Problèmes d'interface Streamlit
------------------------------

Application Streamlit ne démarre pas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Erreur** :

.. code-block:: text

   ModuleNotFoundError: No module named 'streamlit'

**Solution** :

.. code-block:: bash

   pip install streamlit

**Erreur de port** :

.. code-block:: text

   OSError: [Errno 48] Address already in use

**Solution** :

.. code-block:: bash

   # Spécifier un port différent
   streamlit run interface/app.py --server.port 8502

Problèmes de visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problème** : Graphiques ne s'affichent pas

**Solutions** :

1. **Installation de Plotly**
   
   .. code-block:: bash
   
      pip install plotly

2. **Vérification des imports**
   
   .. code-block:: python
   
      import plotly.graph_objects as go
      import plotly.express as px

3. **Fallback vers Matplotlib**
   
   .. code-block:: python
   
      try:
           import plotly.graph_objects as go
           USE_PLOTLY = True
       except ImportError:
           import matplotlib.pyplot as plt
           USE_PLOTLY = False

Erreurs de prédiction
~~~~~~~~~~~~~~~~~~~~

**Problème** : Prédictions donnent des valeurs aberrantes

**Diagnostic** :

.. code-block:: python

   # Vérification de la normalisation
   print(f"Min prédiction: {predictions.min()}")
   print(f"Max prédiction: {predictions.max()}")
   print(f"Valeurs NaN: {np.isnan(predictions).sum()}")

**Solutions** :

1. **Vérification de la dénormalisation**
   
   .. code-block:: python
   
      # S'assurer que le scaler est correctement appliqué
      predictions_denorm = scaler.inverse_transform(predictions.reshape(-1, 1))

2. **Clipping des valeurs**
   
   .. code-block:: python
   
      # Limiter les prédictions à des valeurs réalistes
      predictions = np.clip(predictions, 0, 5000)  # 0 à 5000 MW

Problèmes de performance
-----------------------

Entraînement trop lent
~~~~~~~~~~~~~~~~~~~~~

**Solutions** :

1. **Utilisation du GPU**
   
   .. code-block:: python
   
      # Vérifier que le GPU est utilisé
      with tf.device('/GPU:0'):
           model.fit(X_train, y_train)

2. **Optimisation des données**
   
   .. code-block:: python
   
      # Utilisation de tf.data pour optimiser le pipeline
      dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
      dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

3. **Parallélisation**
   
   .. code-block:: python
   
      # Utiliser tous les CPU disponibles
      import multiprocessing
      model.fit(X_train, y_train, workers=multiprocessing.cpu_count())

Prédictions lentes
~~~~~~~~~~~~~~~~~

**Solutions** :

1. **Batch predictions**
   
   .. code-block:: python
   
      # Prédire par batches au lieu d'une par une
      predictions = model.predict(X_test, batch_size=32)

2. **Optimisation du modèle**
   
   .. code-block:: python
   
      # Quantification du modèle
      converter = tf.lite.TFLiteConverter.from_keras_model(model)
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      tflite_model = converter.convert()

Problèmes de déploiement
-----------------------

Problèmes avec requirements.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution** : Créer un requirements.txt précis

.. code-block:: text

   # requirements.txt
   pandas==1.5.3
   numpy==1.24.3
   matplotlib==3.7.1
   seaborn==0.12.2
   scikit-learn==1.3.0
   tensorflow==2.13.0
   streamlit==1.28.1
   plotly==5.17.0
   optuna==3.4.0

Problèmes avec l'environnement virtuel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Créer un nouvel environnement
   python -m venv energy_env
   
   # Windows
   energy_env\Scripts\activate
   
   # Installation propre
   pip install --upgrade pip
   pip install -r requirements.txt

Debugging avancé
---------------

Activation du debugging TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import tensorflow as tf
   
   # Activer le debugging
   tf.debugging.set_log_device_placement(True)
   
   # Logs détaillés
   tf.get_logger().setLevel('INFO')

Profiling des performances
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cProfile
   import pstats
   
   # Profiling de l'entraînement
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Code à profiler
   model.fit(X_train, y_train)
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)

Vérification de l'intégrité des données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_data_integrity(data):
       """Vérification complète des données"""
       
       issues = []
       
       # Vérification des types
       for col in data.columns:
           if data[col].dtype == 'object':
               try:
                   pd.to_numeric(data[col])
               except:
                   issues.append(f"Colonne {col} contient des valeurs non-numériques")
       
       # Vérification des valeurs infinies
       inf_values = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
       if inf_values > 0:
           issues.append(f"{inf_values} valeurs infinies détectées")
       
       # Vérification des doublons
       duplicates = data.duplicated().sum()
       if duplicates > 0:
           issues.append(f"{duplicates} lignes dupliquées")
       
       return issues

Logs et monitoring
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   
   # Configuration des logs
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('energy_prediction.log'),
           logging.StreamHandler()
       ]
   )
   
   logger = logging.getLogger(__name__)
   
   # Utilisation dans le code
   logger.info("Démarrage de l'entraînement")
   logger.warning("Performance dégradée détectée")
   logger.error("Erreur lors de la prédiction")

Contact et support
-----------------

Si les solutions ci-dessus ne résolvent pas votre problème :

1. **Vérifiez la FAQ** : :doc:`faq`
2. **Consultez la documentation API** : :doc:`api_reference`
3. **Créez un issue** sur GitHub avec :
   
   - Description détaillée du problème
   - Code minimal pour reproduire l'erreur
   - Versions des librairies utilisées
   - Messages d'erreur complets

Checklist de dépannage
---------------------

Avant de signaler un problème, vérifiez :

- [ ] Versions des dépendances à jour
- [ ] Chemins des fichiers corrects
- [ ] Données au bon format
- [ ] Modèles pré-entraînés disponibles
- [ ] Mémoire suffisante (RAM/GPU)
- [ ] Permissions de lecture/écriture
- [ ] Variables d'environnement configurées
- [ ] Logs d'erreur consultés

.. note::

   La plupart des problèmes sont liés à des incompatibilités de versions ou des chemins incorrects. Une installation propre dans un environnement virtuel résout souvent les problèmes.
