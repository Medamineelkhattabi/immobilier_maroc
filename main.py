import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from data_generator import generate_real_estate_data
from model import train_and_save_model, load_model, predict_price

# Configuration de la page
st.set_page_config(
    page_title="Analyse du Marché Immobilier Marocain",
    page_icon="🏢",
    layout="wide"
)

# Titre principal
st.title("📊 Analyse du Marché Immobilier Marocain")

# Vérifier si les données existent, sinon les générer
@st.cache_data
def load_data():
    if os.path.exists('immobilier_maroc.csv'):
        df = pd.read_csv('immobilier_maroc.csv')
    else:
        df = generate_real_estate_data(1000)
        df.to_csv('immobilier_maroc.csv', index=False)
    return df

# Chargement des données
df = load_data()

# Vérifier si le modèle existe, sinon l'entraîner
@st.cache_resource
def get_model():
    if os.path.exists('models/immobilier_model.pkl'):
        model = load_model()
    else:
        result = train_and_save_model()
        model = result['model']
    return model

# Chargement du modèle
model = get_model()

# Barre latérale avec menu de navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choisir une section:",
        ["Accueil", "Analyse Descriptive", "Prix par Ville", "Modèle Prédictif"]
    )
    
    st.header("À propos")
    st.info(
        """
        Cette application est un exemple d'analyse du marché immobilier marocain. 
        Les données utilisées sont synthétiques et générées à des fins de démonstration.
        """
    )

# Page d'accueil
if page == "Accueil":
    st.subheader("Bienvenue dans l'application d'analyse du marché immobilier marocain")
    
    st.markdown("""
    Cette application vous permet d'explorer le marché immobilier au Maroc à travers plusieurs fonctionnalités:
    
    1. **Analyse Descriptive**: Statistiques et visualisations sur les données immobilières
    2. **Prix par Ville**: Comparaison des prix entre différentes villes marocaines
    3. **Modèle Prédictif**: Estimation du prix d'un bien immobilier selon ses caractéristiques
    
    *Note: Les données utilisées sont synthétiques et générées à des fins de démonstration.*
    """)
    
    # Quelques statistiques clés
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Prix moyen",
            f"{df['prix'].mean():,.0f} MAD",
            f"{df['prix'].mean() / 10000:.1f} MDH"
        )
    
    with col2:
        st.metric(
            "Prix moyen au m²",
            f"{df['prix_m2'].mean():,.0f} MAD/m²"
        )
    
    with col3:
        st.metric(
            "Superficie moyenne",
            f"{df['superficie'].mean():.0f} m²"
        )
    
    # Aperçu des données
    st.subheader("Aperçu des données")
    st.dataframe(df.head(10))

# Page d'analyse descriptive
elif page == "Analyse Descriptive":
    st.subheader("Analyse Descriptive des Données Immobilières")
    
    # Distribution des types de biens
    st.write("### Distribution des Types de Biens")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='type_bien', ax=ax1)
    ax1.set_title('Nombre de Biens par Type')
    ax1.set_xlabel('Type de Bien')
    ax1.set_ylabel('Nombre')
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    
    # Distribution des prix
    st.write("### Distribution des Prix")
    
    # Option pour filtrer les données
    price_filter = st.slider(
        "Filtrer par prix maximum (MAD)",
        min_value=int(df['prix'].min()),
        max_value=int(df['prix'].max()),
        value=int(df['prix'].quantile(0.95))
    )
    
    filtered_df = df[df['prix'] <= price_filter]
    
    # Histogramme des prix
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=filtered_df, x='prix', bins=30, kde=True, ax=ax2)
    ax2.set_title('Distribution des Prix')
    ax2.set_xlabel('Prix (MAD)')
    ax2.set_ylabel('Fréquence')
    st.pyplot(fig2)
    
    # Relation entre superficie et prix
    st.write("### Relation entre Superficie et Prix")
    
    # Scatterplot avec Plotly
    fig3 = px.scatter(
        filtered_df,
        x='superficie',
        y='prix',
        color='type_bien',
        size='superficie',
        hover_name='ville',
        hover_data=['quartier', 'prix_m2'],
        title='Prix vs Superficie par Type de Bien'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Statistiques descriptives
    st.write("### Statistiques Descriptives")
    
    # Sélection des colonnes numériques
    num_cols = ['prix', 'superficie', 'chambres', 'salles_bain', 'etage', 'age', 'prix_m2']
    st.dataframe(df[num_cols].describe())

# Page des prix par ville
elif page == "Prix par Ville":
    st.subheader("Analyse des Prix par Ville")
    
    # Prix moyen par ville
    st.write("### Prix Moyen par Ville")
    
    ville_stats = df.groupby('ville').agg({
        'prix': 'mean',
        'prix_m2': 'mean',
        'superficie': 'mean'
    }).reset_index().sort_values('prix', ascending=False)
    
    ville_stats = ville_stats.rename(columns={
        'prix': 'Prix Moyen (MAD)',
        'prix_m2': 'Prix Moyen au m² (MAD)',
        'superficie': 'Superficie Moyenne (m²)'
    })
    
    st.dataframe(ville_stats.style.format({
        'Prix Moyen (MAD)': '{:,.0f}',
        'Prix Moyen au m² (MAD)': '{:,.0f}',
        'Superficie Moyenne (m²)': '{:,.1f}'
    }))
    
    # Bar chart des prix moyens par ville
    st.write("### Comparaison des Prix Moyens par Ville")
    
    fig4 = px.bar(
        ville_stats,
        x='ville',
        y='Prix Moyen (MAD)',
        title='Prix Moyen par Ville',
        color='Prix Moyen au m² (MAD)',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Prix par m² par ville
    st.write("### Prix au m² par Ville")
    
    fig5 = px.bar(
        ville_stats,
        x='ville',
        y='Prix Moyen au m² (MAD)',
        title='Prix Moyen au m² par Ville',
        color='Prix Moyen au m² (MAD)',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # Distribution des prix par ville
    st.write("### Distribution des Prix par Ville")
    
    # Sélection de villes
    selected_villes = st.multiselect(
        "Sélectionner des villes à comparer",
        options=df['ville'].unique(),
        default=df['ville'].unique()[:4]
    )
    
    if selected_villes:
        # Box plot des prix par ville
        fig6 = px.box(
            df[df['ville'].isin(selected_villes)],
            x='ville',
            y='prix',
            title='Distribution des Prix par Ville',
            color='ville'
        )
        st.plotly_chart(fig6, use_container_width=True)

# Page du modèle prédictif
elif page == "Modèle Prédictif":
    st.subheader("Estimation du Prix d'un Bien Immobilier")
    
    st.markdown("""
    Utilisez le formulaire ci-dessous pour estimer le prix d'un bien immobilier en fonction de ses caractéristiques.
    """)
    
    # Formulaire de prédiction
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ville = st.selectbox("Ville", sorted(df['ville'].unique()))
            type_bien = st.selectbox("Type de bien", sorted(df['type_bien'].unique()))
            etat = st.selectbox("État du bien", sorted(df['etat'].unique()))
            superficie = st.number_input("Superficie (m²)", min_value=20, max_value=2000, value=100)
            chambres = st.number_input("Nombre de chambres", min_value=0, max_value=10, value=2)
        
        with col2:
            salles_bain = st.number_input("Nombre de salles de bain", min_value=0, max_value=6, value=1)
            etage = st.number_input("Étage", min_value=0, max_value=20, value=0)
            age = st.number_input("Âge du bien (années)", min_value=0, max_value=100, value=5)
            
            ascenseur = st.checkbox("Ascenseur")
            parking = st.checkbox("Parking")
            piscine = st.checkbox("Piscine")
            jardin = st.checkbox("Jardin")
            securite = st.checkbox("Sécurité")
        
        submit_button = st.form_submit_button("Estimer le prix")
    
    # Prédiction
    if submit_button:
        # Préparation des données d'entrée
        input_data = {
            'ville': ville,
            'type_bien': type_bien,
            'etat': etat,
            'superficie': superficie,
            'chambres': chambres,
            'salles_bain': salles_bain,
            'etage': etage,
            'age': age,
            'ascenseur': ascenseur,
            'parking': parking,
            'piscine': piscine,
            'jardin': jardin,
            'securite': securite
        }
        
        # Conversion en DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Prédiction
        predicted_price = predict_price(model, input_df)
        
        # Affichage du résultat
        st.success(f"Prix estimé: **{predicted_price:,.0f} MAD** ({predicted_price/10000:.2f} MDH)")
        
        # Comparaison avec des biens similaires
        st.subheader("Biens similaires")
        
        # Filtrage des biens similaires
        similar_properties = df[
            (df['ville'] == ville) &
            (df['type_bien'] == type_bien) &
            (df['superficie'] >= superficie * 0.8) &
            (df['superficie'] <= superficie * 1.2)
        ].sort_values('prix')
        
        if len(similar_properties) > 0:
            st.dataframe(similar_properties[['ville', 'quartier', 'type_bien', 'superficie', 'chambres', 'salles_bain', 'etat', 'prix']].head(5))
            
            # Statistiques sur les biens similaires
            avg_price = similar_properties['prix'].mean()
            median_price = similar_properties['prix'].median()
            min_price = similar_properties['prix'].min()
            max_price = similar_properties['prix'].max()
            
            st.write(f"Prix moyen des biens similaires: **{avg_price:,.0f} MAD**")
            st.write(f"Prix médian des biens similaires: **{median_price:,.0f} MAD**")
            st.write(f"Fourchette de prix: **{min_price:,.0f}** - **{max_price:,.0f} MAD**")
        else:
            st.info("Aucun bien similaire trouvé dans la base de données.")