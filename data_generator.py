import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_real_estate_data(num_records=1000):
    """
    Génère des données synthétiques pour le marché immobilier marocain
    """
    # Villes marocaines avec leurs prix moyens par m² (en MAD)
    villes = {
        'Casablanca': {'prix_moyen': 12000, 'std': 3000},
        'Rabat': {'prix_moyen': 11000, 'std': 2500},
        'Marrakech': {'prix_moyen': 10000, 'std': 3500},
        'Tanger': {'prix_moyen': 9000, 'std': 2000},
        'Agadir': {'prix_moyen': 8000, 'std': 1800},
        'Fès': {'prix_moyen': 7000, 'std': 1500},
        'Meknès': {'prix_moyen': 6500, 'std': 1400},
        'Oujda': {'prix_moyen': 6000, 'std': 1300}
    }
    
    # Types de biens
    types_bien = ['Appartement', 'Maison', 'Villa', 'Riad', 'Terrain', 'Local commercial']
    
    # Quartiers (génériques)
    quartiers = {
        'Casablanca': ['Maarif', 'Anfa', 'Ain Diab', 'Bourgogne', 'Gauthier', 'Racine', 'CIL'],
        'Rabat': ['Agdal', 'Hassan', 'Hay Riad', 'Souissi', 'Les Orangers'],
        'Marrakech': ['Gueliz', 'Hivernage', 'Palmeraie', 'Médina', 'Amerchich'],
        'Tanger': ['Centre Ville', 'Malabata', 'Cap Spartel', 'Boukhalef'],
        'Agadir': ['Centre', 'Sonaba', 'Founty', 'Talborjt'],
        'Fès': ['Ville Nouvelle', 'Médina', 'Route Immouzer', 'Atlas'],
        'Meknès': ['Hamria', 'Centre Ville', 'Plaisance', 'Marjane'],
        'Oujda': ['Centre Ville', 'Hay El Qods', 'Angad', 'Hay Essalam']
    }
    
    # États du bien
    etats = ['Neuf', 'Bon état', 'À rénover', 'En construction']
    
    # Génération des données
    data = []
    
    # Date actuelle pour référence
    date_actuelle = datetime.now()
    
    for _ in range(num_records):
        # Sélection aléatoire d'une ville
        ville = np.random.choice(list(villes.keys()))
        quartier = np.random.choice(quartiers[ville])
        type_bien = np.random.choice(types_bien)
        etat = np.random.choice(etats)
        
        # La superficie dépend du type de bien
        if type_bien == 'Appartement':
            superficie = np.random.randint(40, 200)
        elif type_bien == 'Maison':
            superficie = np.random.randint(80, 300)
        elif type_bien == 'Villa':
            superficie = np.random.randint(150, 500)
        elif type_bien == 'Riad':
            superficie = np.random.randint(100, 400)
        elif type_bien == 'Terrain':
            superficie = np.random.randint(200, 2000)
        else:  # Local commercial
            superficie = np.random.randint(30, 500)
        
        # Chambres et salles de bain (uniquement pour les biens résidentiels)
        if type_bien in ['Appartement', 'Maison', 'Villa', 'Riad']:
            chambres = np.random.randint(1, 7)
            salles_bain = max(1, min(chambres, np.random.randint(1, 5)))
        else:
            chambres = 0
            salles_bain = 0
        
        # Étage (uniquement pour les appartements)
        if type_bien == 'Appartement':
            etage = np.random.randint(0, 15)
        else:
            etage = 0
        
        # Âge du bien (en années)
        if etat == 'Neuf':
            age = 0
        elif etat == 'En construction':
            age = 0
        else:
            age = np.random.randint(1, 50)
        
        # Prix de base par m²
        prix_base = np.random.normal(villes[ville]['prix_moyen'], villes[ville]['std'])
        
        # Ajustements de prix
        # Facteur type de bien
        if type_bien == 'Villa':
            facteur_type = 1.3
        elif type_bien == 'Riad':
            facteur_type = 1.2
        elif type_bien == 'Maison':
            facteur_type = 1.1
        elif type_bien == 'Local commercial':
            facteur_type = 1.15
        elif type_bien == 'Terrain':
            facteur_type = 0.8
        else:  # Appartement
            facteur_type = 1.0
        
        # Facteur état
        if etat == 'Neuf':
            facteur_etat = 1.2
        elif etat == 'Bon état':
            facteur_etat = 1.0
        elif etat == 'En construction':
            facteur_etat = 0.9
        else:  # À rénover
            facteur_etat = 0.7
        
        # Facteur âge
        facteur_age = max(0.5, 1 - (age * 0.01))
        
        # Prix final
        prix = int(prix_base * superficie * facteur_type * facteur_etat * facteur_age)
        
        # Date de mise en vente (entre il y a 6 mois et aujourd'hui)
        jours_passes = np.random.randint(0, 180)
        date_mise_en_vente = date_actuelle - timedelta(days=jours_passes)
        
        # Ajout des commodités (pour les biens résidentiels)
        if type_bien in ['Appartement', 'Maison', 'Villa', 'Riad']:
            ascenseur = bool(np.random.binomial(1, 0.6)) if type_bien == 'Appartement' and etage > 2 else bool(np.random.binomial(1, 0.1))
            parking = bool(np.random.binomial(1, 0.7))
            piscine = bool(np.random.binomial(1, 0.8)) if type_bien in ['Villa', 'Riad'] else bool(np.random.binomial(1, 0.1))
            jardin = bool(np.random.binomial(1, 0.9)) if type_bien in ['Villa', 'Maison', 'Riad'] else bool(np.random.binomial(1, 0.05))
            securite = bool(np.random.binomial(1, 0.7))
        else:
            ascenseur = False
            parking = bool(np.random.binomial(1, 0.5))
            piscine = False
            jardin = False
            securite = bool(np.random.binomial(1, 0.6))
        
        # Création d'une entrée
        entry = {
            'ville': ville,
            'quartier': quartier,
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
            'securite': securite,
            'prix': prix,
            'prix_m2': int(prix / superficie),
            'date_mise_en_vente': date_mise_en_vente.strftime('%Y-%m-%d')
        }
        
        data.append(entry)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Générer les données
    df = generate_real_estate_data(1000)
    
    # Sauvegarder les données
    df.to_csv('immobilier_maroc.csv', index=False)
    
    print("Données générées avec succès!")
    print(f"Nombre d'enregistrements: {len(df)}")
    print(f"Prix moyen: {df['prix'].mean():.2f} MAD")
    print(f"Prix moyen au m²: {df['prix_m2'].mean():.2f} MAD/m²")