import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="LA Crime Predictor", page_icon="🛡️")

# Importation sécurisée de TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- 2. CHARGEMENT DES FICHIERS ---
@st.cache_resource
def load_all_assets():
    if not TF_AVAILABLE:
        return None, None, None, "TensorFlow n'est pas installé."
    try:
        # LE SECRET : compile=False pour éviter les conflits de versions
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, model_columns, error_msg = load_all_assets()

# --- 3. INTERFACE ---
st.title("🛡️ Analyseur de Risques Criminels - LA")

if error_msg:
    st.error(f"Erreur : {error_msg}")
    st.stop()

# Extraction propre des quartiers depuis model_columns
prefix = 'AREA NAME_'
quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

with st.form("main_form"):
    col1, col2 = st.columns(2)
    area = col1.selectbox("Quartier", options=quartiers)
    age = col1.slider("Âge de la victime", 1, 100, 30)
    hour = col2.number_input("Heure (ex: 2230)", 0, 2359, 1200)
    submit = st.form_submit_button("Lancer la prédiction")

# --- 4. LOGIQUE DE PRÉDICTION (CORRECTION DU "FEATURE NAMES") ---
if submit:
    try:
        # ÉTAPE CRUCIALE : On crée un dictionnaire avec TOUTES les colonnes à 0
        # Cela garantit que le Scaler recevra exactement ce qu'il a vu au fit()
        data_dict = {col: [0.0] for col in model_columns}
        input_df = pd.DataFrame(data_dict)
        
        # Remplissage des valeurs saisies
        if 'Vict Age' in input_df.columns:
            input_df['Vict Age'] = float(age)
        if 'TIME OCC' in input_df.columns:
            input_df['TIME OCC'] = float(hour)
        
        # Activation du quartier choisi (One-Hot Encoding manuel)
        target_col = f"{prefix}{area}"
        if target_col in input_df.columns:
            input_df[target_col] = 1.0
        
        # RÉALIGNEMENT : On force l'ordre des colonnes pour le Scaler
        input_df = input_df[model_columns]

        # Transformation et Prédiction
        X_scaled = scaler.transform(input_df)
        prediction = model.predict(X_scaled)
        
        # --- 5. AFFICHAGE ---
        st.divider()
        p_id = float(prediction[0][0])
        p_agress = float(prediction[0][1])

        c1, c2 = st.columns(2)
        c1.metric("🆔 Vol d'Identité", f"{p_id*100:.1f}%")
        c2.metric("👊 Agression Simple", f"{p_agress*100:.1f}%")

        # Petit graphique de comparaison
        st.bar_chart(pd.DataFrame({
            "Probabilité (%)": [p_id*100, p_agress*100]
        }, index=["Vol Identité", "Agression"]))

    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")
