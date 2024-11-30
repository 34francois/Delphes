import streamlit as st
import pandas as pd
import tensorflow as tf  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tempfile

# Titre de l'application
st.image("/home/francois/Documents/ACROPOLE/Dashboard/dash_1/Delphes_logo.png", width=380)
st.title("Prédiction du risque de décrochage scolaire")

model_file = st.file_uploader("Charger le modèle Keras (formats .keras ou .h5)", type=["keras", "h5"])

# Si un modèle a été chargé
if model_file is not None:
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp_file:
            temp_file.write(model_file.read())
            temp_file_path = temp_file.name

        # Load the model using TensorFlow 2.0 compatible format
        model = tf.keras.models.load_model(temp_file_path) 

        # Chargement du fichier CSV
        uploaded_file = st.file_uploader("Charger le fichier CSV", type=["csv"])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            # Diviser les données en features et target
            X = data[['MOYENNE', 'ABSENCE', 'COMPORTEMENT']]  
            y = data['DECROCHAGE']  

            # Diviser en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Mettre à l'échelle les données
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        else:
            st.warning("Veuillez charger un fichier CSV pour commencer.")

    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")

else:
    st.warning("Veuillez charger un modèle pour commencer.")
