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

# --- 2. CHARGEMENT DES ASSETS ---
@st.cache_resource
def load_all_assets():
    try:
        # Chargement sans compilation pour la compatibilité
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, model_columns, error_msg = load_all_assets()

# --- 3. INTERFACE UTILISATEUR ---
st.title("🛡️ Analyseur de Risques Criminels")
st.subheader("Modèle MLP - Los Angeles")

if error_msg:
    st.error(f"❌ Erreur de chargement : {error_msg}")
    st.stop()

# Extraction des quartiers (ex: 'AREA NAME_Downtown')
prefix = 'AREA NAME_'
quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area_choice = st.selectbox("Quartier", options=quartiers)
        age_input = st.slider("Âge de la victime", 1, 100, 30)
    with col2:
        hour_input = st.number_input("Heure (0-2359)", 0, 2359, 1200)
    
    submit_btn = st.form_submit_button("Lancer l'analyse")

# --- 4. LOGIQUE DE PRÉDICTION ---
if submit_btn:
    try:
        # ÉTAPE CRUCIALE : On crée un DataFrame avec TOUTES les 22 colonnes (remplies de 0)
        # On utilise [0] pour l'index pour créer une seule ligne
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Remplissage des variables numériques (vérifiez bien l'orthographe exacte dans Colab)
        if 'Vict Age' in model_columns:
            input_df['Vict Age'] = age_input
        if 'TIME OCC' in model_columns:
            input_df['TIME OCC'] = hour_input
        
        # Activation du quartier (One-Hot Encoding)
        target_col = f"{prefix}{area_choice}"
        if target_col in model_columns:
            input_df[target_col] = 1

        # TRANSFORMATION : .values garantit que le Scaler reçoit 22 colonnes numériques
        X_scaled = scaler.transform(input_df.values)
        
        # PRÉDICTION
        prediction = model.predict(X_scaled)
        
        # --- 5. AFFICHAGE ---
        st.divider()
        st.success(f"Analyse terminée pour **{area_choice}**")
        
        # On extrait les probabilités (ajustez les index [0] et [1] selon votre modèle)
        prob_vol = float(prediction[0][0]) * 100
        prob_agress = float(prediction[0][1]) * 100

        c1, c2 = st.columns(2)
        c1.metric("🆔 Vol d'Identité", f"{prob_vol:.1f}%")
        c2.metric("👊 Agression Simple", f"{prob_agress:.1f}%")
        
        # Graphique
        st.bar_chart(pd.DataFrame({
            "Crime": ["Vol Identité", "Agression"],
            "Proba %": [prob_vol, prob_agress]
        }).set_index("Crime"))

    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")

st.caption("Déploiement Streamlit Cloud - TensorFlow 2.16+")
