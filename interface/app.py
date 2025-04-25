import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="🔋 Analyse Production vs Consommation", layout="centered")
st.title("🔍 Analyse de la Production vs la Consommation d'Énergie")

# Génération de données aléatoires pour test
nb_jours = st.slider("📅 Choisissez le nombre de jours", min_value=10, max_value=100, value=30)

np.random.seed(42)  # pour reproduire les mêmes résultats
generation = np.random.randint(400, 700, nb_jours)
consommation = np.random.randint(300, 800, nb_jours)
jours = [f"Jour {i+1}" for i in range(nb_jours)]

# Création DataFrame
df = pd.DataFrame({
    "Jour": jours,
    "Énergie Générée": generation,
    "Énergie Consommée": consommation
})

# Analyse
jours_plus_consommation = (df["Énergie Consommée"] > df["Énergie Générée"]).sum()
jours_plus_generation = (df["Énergie Générée"] > df["Énergie Consommée"]).sum()

pourcentage_conso = (jours_plus_consommation / nb_jours) * 100
pourcentage_gene = (jours_plus_generation / nb_jours) * 100

# Affichage
st.subheader("📊 Résultats de l'analyse")
col1, col2 = st.columns(2)
col1.metric("📉 Jours où la consommation > production", jours_plus_consommation, f"{pourcentage_conso:.1f}%")
col2.metric("📈 Jours où la production > consommation", jours_plus_generation, f"{pourcentage_gene:.1f}%")

# Graphique
st.subheader("📈 Comparaison journalière")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(jours, df["Énergie Générée"], label="Production", marker='o')
ax.plot(jours, df["Énergie Consommée"], label="Consommation", marker='x')
ax.set_xlabel("Jours")
ax.set_ylabel("Énergie (kWh)")
ax.set_title("Production vs Consommation")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Afficher les données
st.subheader("📄 Données utilisées")
st.dataframe(df)