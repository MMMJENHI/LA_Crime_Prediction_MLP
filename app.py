if predict_btn:
        # 1. Création d'un DataFrame avec une seule ligne de ZÉROS
        # On utilise EXACTEMENT les colonnes enregistrées dans 'model_columns.pkl'
        final_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # 2. Remplissage des variables numériques
        # On vérifie que les noms correspondent à ceux de ton entraînement
        if 'Vict Age' in model_columns:
            final_df['Vict Age'] = age
        if 'TIME OCC' in model_columns:
            final_df['TIME OCC'] = hour
            
        # 3. Activation du quartier (One-Hot Encoding Manuel)
        # On construit le nom de la colonne comme lors de l'entraînement
        target_col = f'AREA NAME_{area}'
        
        if target_col in model_columns:
            final_df[target_col] = 1
        else:
            st.warning(f"Note : Le quartier '{area}' n'était pas présent lors de l'entraînement initial.")

        # --- LE BOUCLIER FINAL ---
        # On force l'ordre des colonnes pour qu'il soit IDENTIQUE au scaler
        final_df = final_df[model_columns]

        try:
            # 4. Transformation et Prédiction
            X_scaled = scaler.transform(final_df)
            res = model.predict(X_scaled)
            
            # 5. Affichage des résultats
            st.subheader(f"Résultats pour {area}")
            c1, c2 = st.columns(2)
            
            prob_id = float(res[0][0])
            prob_agress = float(res[0][1])

            c1.metric("🆔 Vol d'Identité", f"{prob_id*100:.1f}%")
            c2.metric("👊 Agression Simple", f"{prob_agress*100:.1f}%")
            
            # Graphique
            st.bar_chart(pd.DataFrame({
                "Crime": ["Vol Identité", "Agression"],
                "Probabilité (%)": [prob_id*100, prob_agress*100]
            }).set_index("Crime"))

        except ValueError as e:
            st.error(f"Erreur d'alignement : {e}")
