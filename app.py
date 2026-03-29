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
        model = tf.keras.models.load_model('mon_modele_mlp.h5')
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
        
        # 5. Affichage
        st.subheader(f"Résultats pour {area}")
        c1, c2 = st.columns(2)
        
        # res[0][0] = ID Theft | res[0][1] = Agression
        c1.metric("🆔 Vol d'Identité", f"{res[0][0]*100:.1f}%")
        c2.metric("👊 Agression Simple", f"{res[0][1]*100:.1f}%")
        
        # Graphique de probabilité
        st.bar_chart(pd.DataFrame({
            "Crime": ["Vol Identité", "Agression"],
            "Probabilité (%)": [res[0][0]*100, res[0][1]*100]
        }).set_index("Crime"))
else:
    st.error(f"Erreur de fichiers : {model_columns}")
