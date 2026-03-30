import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- 1. CHARGEMENT ---
df = pd.read_csv('/content/drive/MyDrive/Analyzing-Crime-in-Los-Angeles-main/crimes.csv')
df = df[df['Vict Age'] > 0]

# --- 2. RÉÉQUILIBRAGE ---
df_id = df[df['Crm Cd Desc'] == 'THEFT OF IDENTITY']
df_agression = df[df['Crm Cd Desc'] == 'BATTERY - SIMPLE ASSAULT']
df_calme = df[(df['Crm Cd Desc'] != 'THEFT OF IDENTITY') &
              (df['Crm Cd Desc'] != 'BATTERY - SIMPLE ASSAULT')].sample(len(df_id), random_state=42)

df_balanced = pd.concat([df_id, df_agression, df_calme]).sample(frac=1, random_state=42)

# --- 3. PRÉPARATION DE X (ENTRÉES) ---
X_raw = df_balanced[['AREA NAME', 'Vict Age', 'TIME OCC']]
# On garde drop_first=False pour la compatibilité Streamlit
X_encoded = pd.get_dummies(X_raw, columns=['AREA NAME'], drop_first=False)
model_columns = list(X_encoded.columns)
joblib.dump(model_columns, 'model_columns.pkl')

# --- 4. PRÉPARATION DE Y (SORTIES) ---
# DOIT ÊTRE FAIT AVANT LE SPLIT
y1 = (df_balanced['Crm Cd Desc'] == 'THEFT OF IDENTITY').astype(int)
y2 = (df_balanced['Crm Cd Desc'] == 'BATTERY - SIMPLE ASSAULT').astype(int)
Y = np.column_stack((y1, y2))

# --- 5. NORMALISATION ET SPLIT ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
joblib.dump(scaler, 'scaler.pkl')

# Maintenant Y existe, on peut split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)

# --- 6. ARCHITECTURE MLP ---
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(model_columns),)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=25, batch_size=32, validation_split=0.1)

model.save('mon_modele_mlp.h5')

# --- 7. VÉRIFICATION ---
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\n🔥 RAPPORT FINAL - VOL D'IDENTITÉ :\n", classification_report(Y_test[:, 0], y_pred[:, 0]))
print("\n🔥 RAPPORT FINAL - AGRESSION :\n", classification_report(Y_test[:, 1], y_pred[:, 1]))
