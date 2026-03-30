import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="LA Crime Predictor", page_icon="🛡️")

# --- 2. GESTION DE L'IMPORT TENSORFLOW ---
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- 3. CHARGEMENT SÉCURISÉ DES ASSETS ---
@st.cache_resource
def load_all_assets():
    if not TF_AVAILABLE:
        return None, None, None, "TensorFlow n'est pas installé dans l'environnement."
    
    try:
        # CHARGEMENT DU MODÈLE (compile=False pour la compatibilité)
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        
        # CHARGEMENT DU SCALER ET DES COLONNES
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        
        return model, scaler, model_columns, None
    except Exception as e:
        return None, None, None, str(e)

# Exécution du chargement
model, scaler, model_columns, error_msg = load_all_assets()

# --- 4. INTERFACE UTILISATEUR ---
st.title("🛡️ Analyseur de Risques Criminels - Los Angeles")
st.markdown("Analyse prédictive basée sur un réseau de neurones (MLP).")

if error_msg:
    st.error(f"❌ Erreur de chargement : {error_msg}")
    st.info("Vérifiez la présence de mon_modele_mlp.h5, scaler.pkl et model_columns.pkl sur GitHub.")
    st.stop()

# Extraction des noms de quartiers (sans le préfixe AREA NAME_)
prefix = 'AREA NAME_'
quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area_choice = st.selectbox("Quartier de l'incident", options=quartiers)
        age_input = st.slider("Âge de la victime", 1, 100, 30)
    with col2:
        hour_input = st.number_input("Heure (Format HHMM, ex: 2230)", 0, 2359, 1200)
    
    submit_btn = st.form_submit_button("Lancer la prédiction")

# --- 5. LOGIQUE DE PRÉDICTION (LA CORRECTION) ---
if submit_btn:
    try:
        # A. Reconstruction du DataFrame avec TOUTES les colonnes originales à 0
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # B. Remplissage des variables numériques
        if 'Vict Age' in model_columns:
            input_df['Vict Age'] = float(age_input)
        if 'TIME OCC' in model_columns:
            input_df['TIME OCC'] = float(hour_input)
        
        # C. Activation du quartier (One-Hot Encoding)
        target_col = f"{prefix}{area_choice}"
        if target_col in model_columns:
            input_df[target_col] = 1.0
        
        # D. TRANSFORMATION (L'astuce .values pour éviter l'erreur de noms)
        # On passe une matrice NumPy brute au scaler
        X_scaled = scaler.transform(input_df.values) 
        
        # E. PRÉDICTION
        prediction = model.predict(X_scaled)
        
        # --- 6. AFFICHAGE DES RÉSULTATS ---
        st.divider()
        prob_id = float(prediction[0][0])
        prob_agress = float(prediction[0][1])

        res_col1, res_col2 = st.columns(2)
        res_col1.metric("🆔 Vol d'Identité", f"{prob_id*100:.1f}%")
        res_col2.metric("👊 Agression Simple", f"{prob_agress*100:.1f}%")

        # Graphique de comparaison
        chart_data = pd.DataFrame({
            "Probabilité (%)": [prob_id*100, prob_
