import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Configuration
st.set_page_config(page_title="LA Risk Detector", page_icon="🕵️")

@st.cache_resource
def load_assets():
    try:

        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        cols = joblib.load('model_columns.pkl')
        return model, scaler, cols
    except Exception as e:
        return None, None, str(e)

model, scaler, model_columns = load_assets()

# --- INTERFACE ---
st.title("🛡️ Analyseur de Risques Criminels - LA")
st.markdown("Estimation basée sur l'IA (Modèle MLP Multi-label)")

if model is not None:
    # On extrait les quartiers du fichier .pkl
    quartiers = sorted([c.replace('AREA NAME_', '') for c in model_columns if 'AREA NAME_' in c])

    with st.sidebar:
        st.header("Paramètres")
        area = st.selectbox("Quartier", options=quartiers)
        age = st.slider("Âge de la victime", 1, 100, 30)
        hour = st.number_input("Heure (0000 à 2359)", 0, 2359, 1200)
        predict_btn = st.button("Calculer les risques")

    if predict_btn:
        # 1. Création DataFrame
        input_df = pd.DataFrame([[area, age, hour]], columns=['AREA NAME', 'Vict Age', 'TIME OCC'])

        # 2. Encodage (doit correspondre au drop_first=True de ton entraînement)
        input_encoded = pd.get_dummies(input_df, columns=['AREA NAME'])

        # 3. Alignement sur les colonnes d'entraînement
        final_df = pd.DataFrame(columns=model_columns).fillna(0)
        final_df = pd.concat([final_df, input_encoded]).fillna(0)
        final_df = final_df[model_columns]

        # 4. Prédiction
        X_scaled = scaler.transform(final_df)
        res = model.predict(X_scaled)



    ###


    # --- REMPLACE LA PARTIE AFFICHAGE PAR CELLE-CI ---
        st.subheader(f"Résultats pour {area}")

        prob_id = res[0][0] * 100
        prob_agression = res[0][1] * 100

        # Affichage en texte simple au cas où le JS plante
        st.write(f"### 🆔 Vol d'Identité : {prob_id:.1f}%")
        st.write(f"### 👊 Agression Simple : {prob_agression:.1f}%")

        # Un simple tableau de données (plus léger que des graphiques)
        st.table(pd.DataFrame({
            "Type de Crime": ["Vol d'Identité", "Agression"],
            "Probabilité (%)": [f"{prob_id:.1f}%", f"{prob_id:.1f}%"]
        }))
