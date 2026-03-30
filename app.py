import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="LA Crime Predictor",
    page_icon="🛡️",
    layout="centered"
)

# --- 2. CHARGEMENT SÉCURISÉ DES ASSETS ---
@st.cache_resource
def load_all_assets():
    try:
        # Charger le modèle (sans compilation pour éviter les bugs de version)
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        # Charger le scaler et la liste des colonnes (les 22 features)
        scaler = joblib.load('scaler.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, scaler, model_cols, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, model_columns, error_msg = load_all_assets()

# --- 3. INTERFACE UTILISATEUR ---
st.title("🛡️ Analyseur de Risques Criminels - LA")
st.markdown("Prédiction par Intelligence Artificielle (Modèle MLP)")

if error_msg:
    st.error(f"Erreur de chargement : {error_msg}")
    st.info("Vérifiez que vos fichiers .h5 et .pkl sont bien à la racine de votre GitHub.")
    st.stop()

# Nettoyage des noms de quartiers pour le menu
prefix = 'AREA NAME_'
quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

# --- 4. FORMULAIRE (CORRECTEMENT STRUCTURÉ) ---
with st.form("crime_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        area_choice = st.selectbox("Sélectionnez le Quartier", options=quartiers)
        age_input = st.slider("Âge de la victime", 1, 100, 30)

    with col2:
        hour_input = st.number_input("Heure de l'incident (0-2359)", 0, 2359, 1200)

    # LE BOUTON : Doit impérativement être DANS le bloc 'with st.form'
    submit_btn = st.form_submit_button("🛡️ Lancer l'analyse des risques")

# --- 5. LOGIQUE DE PRÉDICTION ---
if submit_btn:
    try:
        # A. Création du DataFrame avec les 22 colonnes (toutes à 0.0)
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

        # B. Remplissage des données utilisateur
        if 'Vict Age' in model_columns:
            input_df['Vict Age'] = float(age_input)
        if 'TIME OCC' in model_columns:
            input_df['TIME OCC'] = float(hour_input)

        # Activation du quartier (One-Hot Encoding)
        target_col = f"{prefix}{area_choice}"
        if target_col in model_columns:
            input_df[target_col] = 1.0

        # C. Transformation (Utilisation de .values pour éviter les erreurs de noms)
        X_scaled = scaler.transform(input_df.values)

        # D. Prédiction
        prediction = model.predict(X_scaled)

        # E. Affichage des résultats
        st.divider()
        prob_id = float(prediction[0][0]) * 100
        prob_agress = float(prediction[0][1]) * 100

        res_col1, res_col2 = st.columns(2)
        res_col1.metric("🆔 Risque Vol d'Identité", f"{prob_id:.1f}%")
        res_col2.metric("👊 Risque Agression Simple", f"{prob_agress:.1f}%")

        # Graphique visuel
        chart_data = pd.DataFrame({
            "Probabilité (%)": [prob_id, prob_agress]
        }, index=["Vol d'Identité", "Agression Simple"])
        st.bar_chart(chart_data)

    except Exception as e:
        st.error(f"Une erreur technique est survenue : {e}")

# --- PIED DE PAGE ---
st.caption("Projet IA Los Angeles - Déploiement Streamlit Cloud / GitHub")
