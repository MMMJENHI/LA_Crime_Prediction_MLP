import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="LA Crime Predictor",
    page_icon="🛡️",
    layout="centered"
)

# --- 2. GESTION DE L'IMPORT TENSORFLOW ---
# On essaie d'importer TensorFlow de manière sécurisée
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- 3. CHARGEMENT DU MODÈLE ET DES ASSETS ---
@st.cache_resource
def load_all_assets():
    """Charge le modèle MLP, le Scaler et la liste des colonnes"""
    if not TF_AVAILABLE:
        return None, None, None, "Erreur : TensorFlow n'est pas installé dans l'environnement."
    
    try:
        # Ligne CRUCIALE : Chargement sans compilation pour la compatibilité Cloud
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        
        # Chargement des outils de preprocessing sauvegardés sur Colab
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        
        return model, scaler, model_columns, None
    except Exception as e:
        return None, None, None, str(e)

# Exécution du chargement au démarrage
model, scaler, model_columns, error_msg = load_all_assets()

# --- 4. INTERFACE UTILISATEUR ---
st.title("🛡️ Analyseur de Risques Criminels")
st.subheader("Étude prédictive - Los Angeles (MLP Model)")

if error_msg:
    st.error(f"❌ Impossible de charger les fichiers : {error_msg}")
    st.info("Vérifiez que mon_modele_mlp.h5, scaler.pkl et model_columns.pkl sont à la racine de votre GitHub.")
    st.stop()

# Extraction propre de la liste des quartiers à partir des colonnes One-Hot
prefix = 'AREA NAME_'
quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

# Formulaire de saisie
with st.form("prediction_form"):
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        area_choice = st.selectbox("Sélectionnez le Quartier", options=quartiers)
        age_input = st.slider("Âge de la victime", 1, 100, 30)
    
    with col_input2:
        hour_input = st.number_input("Heure de l'incident (0-2359)", 0, 2359, 1200)
    
    submit_btn = st.form_submit_button("Lancer l'analyse")

# --- 5. LOGIQUE DE PRÉDICTION ---
if submit_btn:
    try:
        # Création d'un DataFrame vide avec TOUTES les colonnes attendues par le modèle
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Remplissage des variables numériques
        if 'Vict Age' in model_columns:
            input_df['Vict Age'] = age_input
        if 'TIME OCC' in model_columns:
            input_df['TIME OCC'] = hour_input
        
        # Activation du quartier (One-Hot Encoding manuel)
        target_col = f"{prefix}{area_choice}"
        if target_col in model_columns:
            input_df[target_col] = 1
        
        # Transformation par le Scaler
        X_scaled = scaler.transform(input_df)
        
        # Prédiction avec le modèle Keras
        prediction = model.predict(X_scaled)
        
        # --- 6. AFFICHAGE DES RÉSULTATS ---
        st.divider()
        st.success(f"Résultats pour le quartier : **{area_choice}**")
        
        prob_vol = float(prediction[0][0])
        prob_agress = float(prediction[0][1])

        res_col1, res_col2 = st.columns(2)
        res_col1.metric("🆔 Risque Vol d'Identité", f"{prob_vol*100:.1f}%")
        res_col2.metric("👊 Risque Agression Simple", f"{prob_agress*100:.1f}%")
        
        # Graphique visuel
        chart_data = pd.DataFrame({
            "Type de Crime": ["Vol d'Identité", "Agression"],
            "Probabilité (%)": [prob_vol*100, prob_agress*100]
        }).set_index("Type de Crime")
        
        st.bar_chart(chart_data)

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {e}")

# --- PIED DE PAGE ---
st.caption("Projet IA - Analyse de données Los Angeles - Déploiement Streamlit Cloud")
