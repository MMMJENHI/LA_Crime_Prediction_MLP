# 🛡️ Analyseur de Risques Criminels - Los Angeles

# 🛡️ Analyseur de Risques - Los Angeles

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://la-crime-prediction.streamlit.app/)





## 📋 Présentation du Projet
Ce projet est un **Système Expert de Prédiction de la Criminalité**. Il permet d'estimer les risques et les types de délits probables dans les quartiers de Los Angeles à partir de données historiques du LAPD. L'application utilise l'**Intelligence Artificielle (Deep Learning)** pour identifier des patterns spatio-temporels et fournir une aide à la décision pour la sécurité urbaine.

---

## 🚀 Accès Rapide
Vous pouvez tester l'application de deux manières :

### 💻 Sur Ordinateur
Cliquez sur le bouton "Open in Streamlit" en haut de la page ou sur ce lien :  
👉 **[Lancer l'Analyseur de Risques (Streamlit)](https://la-crime-prediction-mmmjenhi.streamlit.app/)**

### 📱 Sur Mobile (Scan me!)
Scannez ce QR Code avec votre téléphone pour accéder instantanément à l'IA :

<img src="qr_code_la_crime.png" width="250" alt="QR Code">

---

## ✨ Fonctionnalités
* **Analyse Multi-label** : Classification simultanée de plusieurs types de crimes (Vol, Agression, etc.).
* **Estimation de Probabilités** : Calcul précis du niveau de risque pour chaque catégorie de délit.
* **Interface Interactive** : Sélection dynamique du quartier (AREA), de l'heure et du profil de la victime.
* **Diagnostic IA** : Prédiction instantanée via un modèle de neurones MLP (Multi-Layer Perceptron).

## 🛠️ Architecture Technique
L'application suit un pipeline de données rigoureux :

1. **Acquisition** : Saisie des paramètres via l'interface Streamlit.
2. **Prétraitement** : Encodage des variables et normalisation via le fichier **scaler.pkl**.
3. **Extraction** : Alignement des caractéristiques selon la structure **model_columns.pkl**.
4. **Verdict** : Inférence et calcul des scores via le modèle **mon_modele_mlp.h5** (Keras/TensorFlow).

## 📁 Structure du dépôt
* `app.py` : Code principal de l'interface Streamlit.
* `mon_modele_mlp.h5` : Le cerveau de l'IA (modèle entraîné).
* `scaler.pkl` : Fichier de normalisation des données.
* `model_columns.pkl` : Structure des colonnes du modèle.
* `requirements.txt` : Liste des bibliothèques nécessaires.
* `qr_code_la_crime.png` : Image pour l'accès mobile.

## 👷 Auteur
**MMMJENHI** - Développement et Intégration IA
# LA_Crime_Prediction_MLP
