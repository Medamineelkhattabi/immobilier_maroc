import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

def prepare_data(df):
    """
    Prépare les données pour le modèle prédictif
    """
    # Sélection des features
    features = [
        'ville', 'type_bien', 'etat', 'superficie', 'chambres', 
        'salles_bain', 'etage', 'age', 'ascenseur', 'parking', 
        'piscine', 'jardin', 'securite'
    ]
    
    # Target
    target = 'prix'
    
    # Séparation des features et de la target
    X = df[features]
    y = df[target]
    
    return X, y

def build_model(X, y):
    """
    Construit et entraîne un modèle prédictif pour les prix immobiliers
    """
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Définition des colonnes catégorielles et numériques
    categorical_features = ['ville', 'type_bien', 'etat']
    binary_features = ['ascenseur', 'parking', 'piscine', 'jardin', 'securite']
    numerical_features = ['superficie', 'chambres', 'salles_bain', 'etage', 'age']
    
    # Préprocesseur pour les features catégorielles
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Préprocesseur pour les features numériques
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combinaison des préprocesseurs
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features),
            ('bin', 'passthrough', binary_features)
        ])
    
    # Pipeline complet avec le préprocesseur et le modèle
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Évaluation du modèle
    y_pred = model.predict(X_test)
    
    # Métriques
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Résultats
    result = {
        'model': model,
        'metrics': {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        },
        'test_data': {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    }
    
    return result

def train_and_save_model(data_path='immobilier_maroc.csv'):
    """
    Entraîne et sauvegarde le modèle
    """
    # Chargement des données
    df = pd.read_csv(data_path)
    
    # Préparation des données
    X, y = prepare_data(df)
    
    # Construction du modèle
    result = build_model(X, y)
    
    # Création du dossier models s'il n'existe pas
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Sauvegarde du modèle
    with open('models/immobilier_model.pkl', 'wb') as f:
        pickle.dump(result['model'], f)
    
    return result

def load_model(model_path='models/immobilier_model.pkl'):
    """
    Charge le modèle sauvegardé
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def predict_price(model, input_data):
    """
    Prédit le prix d'un bien immobilier
    """
    # Conversion en dataframe si nécessaire
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    
    # Prédiction
    prediction = model.predict(input_data)[0]
    
    return prediction

if __name__ == "__main__":
    # Entraînement et sauvegarde du modèle
    result = train_and_save_model()
    
    # Affichage des résultats
    print("Modèle entraîné avec succès!")
    print("Métriques:")
    for metric, value in result['metrics'].items():
        print(f"{metric}: {value:.2f}")