import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# --- CONFIGURATION ET COMPATIBILITÉ ---
st.set_page_config(page_title="LA Crime Detector", page_icon="🛡️")
os.environ['TF_USE_LEGACY_KERAS'] = '1'

@st.cache_resource
def load_assets():
    """Chargement sécurisé du modèle et des outils de preprocessing"""
    try:
        # On charge le modèle sans compiler pour éviter les conflits de version
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except Exception as e:
        return None, None, str(e)

model, scaler, model_columns = load_assets()

# --- INTERFACE UTILISATEUR ---
st.title("🛡️ Analyseur de Risques Criminels - Los Angeles")
st.markdown("Cette application utilise un modèle de deep learning (MLP) pour estimer les probabilités de types de crimes.")

if model is not None:
    # Extraction des noms de quartiers à partir des colonnes d'entraînement
    prefix = 'AREA NAME_'
    quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

    with st.sidebar:
        st.header("Paramètres de l'analyse")
        area = st.selectbox("Quartier de l'incident", options=quartiers)
        age = st.slider("Âge de la victime", 1, 100, 25)
        hour = st.number_input("Heure (Format HHMM, ex: 2230)", 0, 2359, 1200)
        predict_btn = st.button("Lancer la prédiction")

    if predict_btn:
        # --- ALIGNEMENT STRICT DES DONNÉES ---
        # 1. Création d'un tableau vide avec TOUTES les colonnes connues du modèle
        final_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # 2. Remplissage des données numériques
        if 'Vict Age' in model_columns: final_df['Vict Age'] = age
        if 'TIME OCC' in model_columns: final_df['TIME OCC'] = hour
            
        # 3. Activation du quartier choisi (One-Hot Encoding)
        target_col = f'AREA NAME_{area}'
        if target_col in model_columns:
            final_df[target_col] = 1

        # 4. Prédiction
        try:
            X_scaled = scaler.transform(final_df)
            res = model.predict(X_scaled)
            
            # --- AFFICHAGE DES RÉSULTATS ---
            st.divider()
            st.subheader(f"Estimation pour le quartier : {area}")
            
            prob_id = float(res[0][0])
            prob_agress = float(res[0][1])

            col1, col2 = st.columns(2)
            col1.metric("🆔 Vol d'Identité", f"{prob_id*100:.1f}%")
            col2.metric("👊 Agression Simple", f"{prob_agress*100:.1f}%")

            # Graphique de comparaison
            chart_data = pd.DataFrame({
                "Crime": ["Vol Identité", "Agression"],
                "Probabilité (%)": [prob_id*100, prob_agress*100]
            }).set_index("Crime")
            st.bar_chart(chart_data)

        except Exception as e:
            st.error(f"Erreur technique lors du calcul : {e}")
else:
    st.error(f"❌ Impossible de charger les fichiers du modèle : {model_columns}")
    st.info("Vérifiez que mon_modele_mlp.h5, scaler.pkl et model_columns.pkl sont bien à la racine de votre GitHub.")
