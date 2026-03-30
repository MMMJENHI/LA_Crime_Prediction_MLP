import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import qrcode
from io import BytesIO

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="LA Crime Predictor", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_all_assets():
    try:
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, scaler, model_cols, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, model_columns, error_msg = load_all_assets()

# --- 2. INTERFACE ---
st.title("🛡️ Analyseur de Risques Criminels - LA")
st.markdown("Prédiction par Intelligence Artificielle (Modèle MLP)")

if error_msg:
    st.error(f"Erreur de chargement : {error_msg}")
    st.stop()

prefix = 'AREA NAME_'
quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

# --- 3. FORMULAIRE ---
with st.form("crime_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area_choice = st.selectbox("Sélectionnez le Quartier", options=quartiers)
        age_input = st.slider("Âge de la victime", 1, 100, 30)
    with col2:
        hour_input = st.number_input("Heure de l'incident (0-2359)", 0, 2359, 1200)
    
    submit_btn = st.form_submit_button("🛡️ Lancer l'analyse des risques")

# --- 4. LOGIQUE DE PRÉDICTION ET CONSEILS (DANS LE MÊME BLOC) ---
if submit_btn:
    try:
        # A. Préparation
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        if 'Vict Age' in model_columns: input_df['Vict Age'] = float(age_input)
        if 'TIME OCC' in model_columns: input_df['TIME OCC'] = float(hour_input)
        target_col = f"{prefix}{area_choice}"
        if target_col in model_columns: input_df[target_col] = 1.0

        # B. Calcul
        X_scaled = scaler.transform(input_df.values)
        prediction = model.predict(X_scaled)
        prob_id = float(prediction[0][0]) * 100
        prob_agress = float(prediction[0][1]) * 100

        # C. Affichage Résultats
        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("🆔 Risque Vol d'Identité", f"{prob_id:.1f}%")
        res_col2.metric("👊 Risque Agression Simple", f"{prob_agress:.1f}%")

        # D. Conseils de Sécurité (Sécurisés contre le NameError)
        st.subheader("💡 Conseils de Sécurité")
        if prob_id > 50:
            st.warning("**Alerte Vol d'Identité :** Risque élevé. Surveillez vos comptes bancaires.")
        if prob_agress > 50:
            st.error("**Alerte Sécurité Physique :** Risque d'agression. Restez vigilant la nuit.")
        if prob_id <= 50 and prob_agress <= 50:
            st.success("Indicateurs de risques modérés. Maintenez une vigilance standard.")

    except Exception as e:
        st.error(f"Erreur : {e}")

# --- 5. SECTION EXPERT ET QR CODE (TOUJOURS VISIBLES) ---
st.divider()
with st.expander("📊 Détails techniques et Partage"):
    st.subheader("Architecture MLP")
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    st.code("\n".join(stringlist))
    
    st.divider()
    
    # GÉNÉRATION QR CODE
    st.subheader("📲 Partager l'application")
    url_app = "https://la-crime-prediction-mlp.streamlit.app/" # À mettre à jour si besoin
    qr = qrcode.make(url_app)
    buf = BytesIO()
    qr.save(buf, format="PNG")
    st.image(buf, caption="Scannez pour tester sur mobile", width=150)

st.caption("Projet IA Los Angeles - Déploiement Streamlit Cloud / GitHub")
