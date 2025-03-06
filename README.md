# Analyse des Prix Immobiliers au Maroc

## Description
Ce projet vise à analyser le marché immobilier marocain en utilisant des techniques de data science et de machine learning. Il inclut une interface interactive via **Streamlit**, permettant aux utilisateurs d'explorer les données, visualiser les tendances des prix et prédire la valeur des biens immobiliers.

## Fonctionnalités
- **Génération de données** : Simulation de données synthétiques avec des variations réalistes
- **Analyse descriptive** : Exploration des données immobilières
- **Visualisation interactive** : Prix par ville et type de bien
- **Modélisation prédictive** : Prédiction du prix des biens immobiliers
- **Dashboard interactif** : Interface utilisateur via Streamlit

## Technologies Utilisées
- **Python**
- **Pandas** : Manipulation de données
- **Scikit-learn** : Modélisation et prédiction
- **Matplotlib & Seaborn** : Visualisation des données
- **Streamlit** : Dashboard interactif

## Installation

1. **Cloner le répertoire**
   ```bash
   git clone https://github.com/votre-utilisateur/immobilier_analyser.git
   cd immobilier_analyser
   ```

2. **Créer un environnement virtuel et l'activer**
   ```bash
   python -m venv immobilier_env
   source immobilier_env/bin/activate  # Pour macOS/Linux
   immobilier_env\Scripts\activate  # Pour Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer l'application Streamlit**
   ```bash
   streamlit run main.py
   ```

## Structure du Projet
```
immobilier_analyser/
│── data/                   # Dossiers contenant les données
│── scripts/                # Scripts de préparation et d'analyse
│── models/                 # Modèles de machine learning
│── main.py                 # Point d'entrée de l'application
│── requirements.txt        # Liste des dépendances
│── README.md               # Documentation du projet
```


## Licence
Ce projet est sous licence MIT.



