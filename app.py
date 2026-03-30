import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Configuration de la page
st.set_page_config(page_title="Crime Detector LA", layout="centered")

@st.cache_resource
def load_all_models():
    """Charge les fichiers sans passer par le CSV"""
    try:
        m = tf.keras.models.load_model('mon_modele_mlp.h5')
        s = joblib.load('scaler.pkl')
        c = joblib.load('model_columns.pkl')
        return m, s, c
    except Exception as e:
        st.error(f"Erreur de fichiers : {e}")
        return None, None, None

model, scaler, model_columns = load_all_models()

if model is not None:
    st.title("🛡️ Analyseur de Risques - Los Angeles")

    # On récupère les quartiers depuis model_columns (Pas besoin du CSV !)
    quartiers = sorted([name.replace('AREA NAME_', '') for name in model_columns if 'AREA NAME_' in name])

    with st.form("main_form"):
        area = st.selectbox("Quartier", quartiers)
        age = st.slider("Age de la personne", 1, 100, 25)
        hour = st.number_input("Heure (0-2359)", 0, 2359, 1200)
        submit = st.form_submit_button("Lancer la prédiction")

    if submit:
        # 1. Création du DataFrame
        input_data = pd.DataFrame([[area, age, hour]], columns=['AREA NAME', 'Vict Age', 'TIME OCC'])

        # 2. Encodage
        input_encoded = pd.get_dummies(input_data, columns=['AREA NAME'])

        # 3. Alignement (Reconstruction du vecteur X)
        final_df = pd.DataFrame(columns=model_columns).fillna(0)
        final_df = pd.concat([final_df, input_encoded]).fillna(0)
        final_df = final_df[model_columns]

        # 4. Prédiction
        pred = model.predict(scaler.transform(final_df))

        # 5. Affichage
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("🆔 Risque Vol Identité", f"{pred[0][0]*100:.1f}%")
        col2.metric("👊 Risque Agression", f"{pred[0][1]*100:.1f}%")
else:
    st.warning("⚠️ Les fichiers .h5 ou .pkl sont manquants dans le dossier Colab.")
