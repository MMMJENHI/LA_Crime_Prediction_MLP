import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="LA Crime Predictor", page_icon="🛡️")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- 2. CHARGEMENT (Modèle + Scaler + Liste des 22 colonnes) ---
@st.cache_resource
def load_assets():
    if not TF_AVAILABLE:
        return None, None, None, "TensorFlow non installé."
    try:
        # On charge les 3 fichiers indispensables
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        model_cols = joblib.load('model_columns.pkl') # C'est ici que sont les 22 noms
        return model, scaler, model_cols, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, model_columns, error_msg = load_assets()

# --- 3. INTERFACE ---
st.title("🛡️ Analyseur de Risques Criminels - LA")

if error_msg:
    st.error(f"Erreur de chargement : {error_msg}")
    st.stop()

# On extrait les quartiers pour le menu déroulant
prefix = 'AREA NAME_'
quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

with st.form("crime_form"):
    col1, col2 = st.columns(2)
    area_choice = col1.selectbox("Quartier", options=quartiers)
    age_input = col1.slider("Âge de la victime", 1, 100, 30)
    hour_input = col2.number_input("Heure (HHMM, ex: 2230)", 0, 2359, 1200)
    submit_btn = st.form_submit_button("Lancer la prédiction")

# --- 4. LOGIQUE DE PRÉDICTION (LA CORRECTION DES 22 FEATURES) ---
if submit_btn:
    try:
        # ÉTAPE CLÉ : On crée un DataFrame de ZÉROS avec les 22 colonnes exactes
        # Cela garantit que le Scaler recevra 22 features et non 5.
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # On remplit l'âge et l'heure (si présents dans les 22 colonnes)
        if 'Vict Age' in model_columns:
            input_df['Vict Age'] = float(age_input)
        if 'TIME OCC' in model_columns:
            input_df['TIME OCC'] = float(hour_input)
        
        # On active le quartier sélectionné (Met un 1 dans la bonne colonne)
        target_col = f"{prefix}{area_choice}"
        if target_col in model_columns:
            input_df[target_col] = 1.0
        
        # TRANSFORMATION : On envoie les 22 colonnes au scaler
        # .values transforme le DataFrame en matrice NumPy pour ignorer les noms
        X_scaled = scaler.transform(input_df.values)
        
        # PRÉDICTION
        prediction = model.predict(X_scaled)
        
        # --- 5. AFFICHAGE ---
        st.divider()
        prob_id = float(prediction[0][0]) * 100
        prob_agress = float(prediction[0][1]) * 100

        res1, res2 = st.columns(2)
        res1.metric("🆔 Vol d'Identité", f"{prob_id:.1f}%")
        res2.metric("👊 Agression Simple", f"{prob_agress:.1f}%")
        
        # Graphique simple
        st.bar_chart(pd.DataFrame({
            "Probabilité (%)": [prob_id, prob_agress]
        }, index=["Vol Identité", "Agression"]))

    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")
