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

# --- 2. CHARGEMENT DES ASSETS (MODÈLE + SCALER + COLONNES) ---
@st.cache_resource
def load_all_assets():
    try:
        # Chargement du modèle Keras (sans compilation pour éviter les conflits)
        model = tf.keras.models.load_model('mon_modele_mlp.h5', compile=False)
        
        # Chargement du scaler et de la liste exacte des colonnes d'entraînement (22 colonnes)
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        
        return model, scaler, model_columns, None
    except Exception as e:
        return None, None, None, str(e)

# Exécution du chargement
model, scaler, model_columns, error_msg = load_all_assets()

# --- 3. INTERFACE UTILISATEUR ---
st.title("🛡️ Analyseur de Risques Criminels")
st.subheader("Étude prédictive - Los Angeles (MLP)")

if error_msg:
    st.error(f"❌ Erreur de chargement : {error_msg}")
    st.info("Vérifiez que mon_modele_mlp.h5, scaler.pkl et model_columns.pkl sont sur GitHub.")
    st.stop()

# Extraction des noms de quartiers à partir des colonnes One-Hot (ex: AREA NAME_Central)
prefix = 'AREA NAME_'
quartiers = sorted([c.replace(prefix, '') for c in model_columns if c.startswith(prefix)])

# Formulaire de saisie
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area_choice = st.selectbox("Sélectionnez le Quartier", options=quartiers)
        age_input = st.slider("Âge de la victime", 1, 100, 30)
    with col2:
        hour_input = st.number_input("Heure de l'incident (Format HHMM, ex: 1430)", 0, 2359, 1200)
    
    submit_btn = st.form_submit_button("Lancer l'analyse prédictive")

# --- 4. LOGIQUE DE PRÉDICTION ---
if submit_btn:
    try:
        # ÉTAPE CRUCIALE : Créer un DataFrame avec TOUTES les 22 colonnes (remplies de 0)
        # Cela garantit que le Scaler recevra le bon nombre de "features"
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Remplissage des variables numériques
        if 'Vict Age' in model_columns:
            input_df['Vict Age'] = age_input
        if 'TIME OCC' in model_columns:
            input_df['TIME OCC'] = hour_input
        
        # Activation du quartier sélectionné (Met la colonne correspondante à 1)
        target_col = f"{prefix}{area_choice}"
        if target_col in model_columns:
            input_df[target_col] = 1
        
        # TRANSFORMATION : On utilise .values pour envoyer uniquement les données numériques
        # Cela règle l'erreur "X has 5 features, but StandardScaler is expecting 22"
        X_scaled = scaler.transform(input_df.values)
        
        # PRÉDICTION
        prediction = model.predict(X_scaled)
        
        # --- 5. AFFICHAGE DES RÉSULTATS ---
        st.divider()
        st.success(f"Analyse terminée pour le quartier : **{area_choice}**")
        
        # On suppose que prediction[0][0] = Vol d'Identité et prediction[0][1] = Agression
        prob_vol = float(prediction[0][0]) * 100
        prob_agress = float(prediction[0][1]) * 100

        res_c1, res_c2 = st.columns(2)
        res_c1.metric("🆔 Risque Vol d'Identité", f"{prob_vol:.1f}%")
        res_c2.metric("👊 Risque Agression Simple", f"{prob_agress:.1f}%")
        
        # Graphique visuel
        chart_data = pd.DataFrame({
            "Type de Crime": ["Vol d'Identité", "Agression"],
            "Probabilité (%)": [prob_vol, prob_agress]
        }).set_index("Type de Crime")
        
        st.bar_chart(chart_data)

    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul : {e}")

st.caption("Données basées sur les archives LAPD - Déploiement Streamlit Cloud")
