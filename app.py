import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# --- CORRECTIF DE COMPATIBILITÉ KERAS ---
# Force l'ancien moteur pour éviter l'erreur 'batch_shape' sur Streamlit Cloud
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Configuration de la page
st.set_page_config(page_title="LA Risk Detector", page_icon="🕵️")

@st.cache_resource
def load_assets():
    try:
        # Chargement sans compilation pour ignorer les erreurs de config d'entraînement
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        cols = joblib.load('model_columns.pkl')
        return model, scaler, cols
    except Exception as e:
        return None, None, str(e)

model, scaler, model_columns = load_assets()

# --- INTERFACE ---
st.title("🛡️ Analyseur de Risques Criminels - LA")
st.markdown("Estimation basée sur l'IA (Modèle MLP Multi-label)")

if model is not None:
    # On extrait les noms des quartiers à partir de la liste des colonnes du modèle
    # On cherche tout ce qui commence par 'AREA NAME_'
    prefix = 'AREA NAME_'
    quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

    with st.sidebar:
        st.header("Paramètres")
        area = st.selectbox("Quartier", options=quartiers)
        age = st.slider("Âge de la victime", 1, 100, 30)
        hour = st.number_input("Heure (Format HHMM, ex: 1430)", 0, 2359, 1200)
        predict_btn = st.button("Calculer les risques")

    if predict_btn:
        # --- ÉTAPE CRUCIALE : ALIGNEMENT DES FEATURES ---
        
        # 1. On crée un DataFrame vide avec TOUTES les colonnes connues par le Scaler
        # Initialisées à 0 (très important pour le One-Hot Encoding)
        final_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # 2. On remplit les valeurs numériques (Vérifie bien l'orthographe exacte dans ton CSV)
        if 'Vict Age' in model_columns:
            final_df['Vict Age'] = age
        if 'TIME OCC' in model_columns:
            final_df['TIME OCC'] = hour
        
        # 3. Activation du quartier (One-Hot Encoding Manuel)
        target_col = f'AREA NAME_{area}'
        if target_col in model_columns:
            final_df[target_col] = 1
        
        # 4. On force l'ORDRE des colonnes (pour éviter l'erreur "Feature names mismatch")
        final_df = final_df[model_columns]

        try:
            # 5. Transformation et Prédiction
            X_scaled = scaler.transform(final_df)
            res = model.predict(X_scaled)
            
            # 6. Affichage des résultats
            st.subheader(f"Résultats pour {area}")
            c1, c2 = st.columns(2)
            
            # Conversion en % (res[0][0] = ID Theft, res[0][1] = Agression)
            prob_id = float(res[0][0])
            prob_agress = float(res[0][1])

            c1.metric("🆔 Vol d'Identité", f"{prob_id*100:.1f}%")
            c2.metric("👊 Agression Simple", f"{prob_agress*100:.1f}%")
            
            # Graphique visuel
            chart_data = pd.DataFrame({
                "Type de Crime": ["Vol d'Identité", "Agression Simple"],
                "Probabilité (%)": [prob_id * 100, prob_agress * 100]
            }).set_index("Type de Crime")
            
            st.bar_chart(chart_data)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.info("Détail technique : Le scaler n'a pas reconnu les colonnes envoyées.")

else:
    st.error(f"❌ Erreur de chargement des fichiers : {model_columns}")
    st.info("Vérifie que mon_modele_mlp.h5, scaler.pkl et model_columns.pkl sont bien sur GitHub.")
