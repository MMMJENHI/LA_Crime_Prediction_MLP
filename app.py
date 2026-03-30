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

# --- 2. CHARGEMENT DES ASSETS (Modèle, Scaler et Liste des 22 colonnes) ---
@st.cache_resource
def load_all_assets():
    if not TF_AVAILABLE:
        return None, None, None, "TensorFlow n'est pas installé."
    try:
        # Charger le modèle MLP (compile=False pour éviter les conflits de version)
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        # Charger le Scaler (celui qui attend 22 colonnes)
        scaler = joblib.load('scaler.pkl')
        # Charger la liste EXACTE des 22 noms de colonnes utilisés sur Colab
        model_cols = joblib.load('model_columns.pkl')
        return model, scaler, model_cols, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, model_columns, error_msg = load_all_assets()

# --- 3. INTERFACE UTILISATEUR ---
st.title("🛡️ Analyseur de Risques Criminels - LA")

if error_msg:
    st.error(f"❌ Erreur de chargement : {error_msg}")
    st.stop()

# On prépare la liste des quartiers pour le menu (en enlevant 'AREA NAME_')
prefix = 'AREA NAME_'
quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        area_choice = st.selectbox("Sélectionnez le Quartier", options=quartiers)
        age_input = st.slider("Âge de la victime", 1, 100, 30)
    with col2:
        hour_input = st.number_input("Heure de l'incident (Format HHMM, ex: 1430)", 0, 2359, 1200)
    
    submit_btn = st.form_submit_button("Lancer l'analyse")

# --- 4. LOGIQUE DE PRÉDICTION (LA RÉPARATION DES 22 COLONNES) ---
if submit_btn:
    try:
        # ÉTAPE CRUCIALE : Création d'un DataFrame de ZÉROS avec les 22 colonnes exactes
        # Cela garantit que le Scaler recevra 22 features et non 5.
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # Remplissage des variables numériques (Age et Heure)
        if 'Vict Age' in model_columns:
            input_df['Vict Age'] = float(age_input)
        if 'TIME OCC' in model_columns:
            input_df['TIME OCC'] = float(hour_input)
        
        # Activation du quartier choisi (One-Hot Encoding manuel)
        target_col = f"{prefix}{area_choice}"
        if target_col in model_columns:
            input_df[target_col] = 1.0
        
        # TRANSFORMATION : On envoie les 22 colonnes au scaler
        # .values transforme le DataFrame en matrice NumPy pour ignorer les noms de colonnes
        X_scaled = scaler.transform(input_df.values)
        
        # PRÉDICTION AVEC LE MODÈLE MLP
        prediction = model.predict(X_scaled)
        
        # --- 5. AFFICHAGE DES RÉSULTATS ---
        st.divider()
        prob_id = float(prediction[0][0]) * 100
        prob_agress = float(prediction[0][1]) * 100

        res1, res2 = st.columns(2)
        res1.metric("🆔 Risque Vol d'Identité", f"{prob_id:.1f}%")
        res2.metric("👊 Risque Agression Simple", f"{prob_agress:.1f}%")
        
        # Graphique visuel
        st.bar_chart(pd.DataFrame({
            "Probabilité (%)": [prob_id, prob_agress]
        }, index=["Vol Identité", "Agression"]))

    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")

st.caption("Modèle MLP v1.0 - Données Los Angeles - Déploiement Streamlit Cloud")
