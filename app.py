import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. GESTION DE L'IMPORT TENSORFLOW ---
try:
    import tensorflow as tf
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# Configuration de la page
st.set_page_config(page_title="LA Crime Detector", page_icon="🛡️")

# --- 2. CHARGEMENT SÉCURISÉ DES ASSETS ---
@st.cache_resource
def load_all_assets():
    if not KERAS_AVAILABLE:
        return None, None, None, "TensorFlow/Keras n'est pas installé. Vérifiez requirements.txt."
    
    try:
        # Chargement du modèle (sans compiler pour éviter les erreurs de version)
        # Par :
        import tf_keras
        model = tf_keras.models.load_model('mon_modele_mlp.h5', compile=False)
        
        # Chargement du scaler et des colonnes
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, model_columns, error_msg = load_all_assets()

# --- 3. INTERFACE UTILISATEUR ---
st.title("🛡️ Analyseur de Risques - Los Angeles")

if error_msg:
    st.error(f"❌ Erreur de chargement : {error_msg}")
    st.info("Vérifiez que mon_modele_mlp.h5, scaler.pkl et model_columns.pkl sont bien à la racine de votre GitHub.")
elif model is not None:
    # Extraction des quartiers (on nettoie les noms issus du One-Hot Encoding)
    quartiers = sorted([c.replace('AREA NAME_', '') for c in model_columns if 'AREA NAME_' in c])

    with st.form("prediction_form"):
        st.subheader("Paramètres de l'incident")
        area = st.selectbox("Quartier", options=quartiers)
        age = st.slider("Âge de la victime", 1, 100, 30)
        hour = st.number_input("Heure (Format HHMM, ex: 2230)", 0, 2359, 1200)
        submit = st.form_submit_button("Calculer les probabilités")

    if submit:
        # --- 4. PRÉPARATION DES DONNÉES (ALIGNEMENT STRICT) ---
        # On crée un DataFrame vide avec toutes les colonnes attendues par le modèle
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Remplissage des variables numériques
        if 'Vict Age' in model_columns: input_df['Vict Age'] = age
        if 'TIME OCC' in model_columns: input_df['TIME OCC'] = hour
        
        # Remplissage de la variable catégorielle (Quartier)
        target_col = f'AREA NAME_{area}'
        if target_col in model_columns:
            input_df[target_col] = 1

        # --- 5. PRÉDICTION ---
        try:
            X_scaled = scaler.transform(input_df)
            prediction = model.predict(X_scaled)
            
            st.divider()
            col1, col2 = st.columns(2)
            
            prob_vol = float(prediction[0][0]) * 100
            prob_agression = float(prediction[0][1]) * 100

            col1.metric("🆔 Vol d'Identité", f"{prob_vol:.1f}%")
            col2.metric("👊 Agression Simple", f"{prob_agression:.1f}%")
            
            # Petit graphique pour le visuel
            st.bar_chart(pd.DataFrame({
                "Risques": ["Vol Identité", "Agression"],
                "Probabilité (%)": [prob_vol, prob_agression]
            }).set_index("Risques"))

        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")
