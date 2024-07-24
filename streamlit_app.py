#C'est le bout de code proprement dit pour le streamlit.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import geojson
import json
from sklearn.preprocessing import MinMaxScaler
import os



# Définir le chemin de base pour les fichiers de données
BASE_PATH = r"E:\expgci\downloads\Donnees-20240724T155917Z-001.zip\Donnees"

@st.cache_data
def load_data():
    # Chargement des fichiers caractéristiques
    carac_2022 = pd.read_csv(os.path.join(BASE_PATH, 'caracteristiques-2022.csv'), header=0, sep=";")
    carac_2021 = pd.read_csv(os.path.join(BASE_PATH, 'caracteristiques-2021.csv'), header=0, sep=";")
    carac_2020 = pd.read_csv(os.path.join(BASE_PATH, 'caracteristiques-2020.csv'), header=0, sep=";")
    carac_2019 = pd.read_csv(os.path.join(BASE_PATH, 'caracteristiques-2019.csv'), header=0, sep=";")

    # Chargement des fichiers usagers
    usagers_2022 = pd.read_csv(os.path.join(BASE_PATH, 'usagers-2022.csv'), header=0, sep=";")
    usagers_2021 = pd.read_csv(os.path.join(BASE_PATH, 'usagers-2021.csv'), header=0, sep=";")
    usagers_2020 = pd.read_csv(os.path.join(BASE_PATH, 'usagers-2020.csv'), header=0, sep=";")
    usagers_2019 = pd.read_csv(os.path.join(BASE_PATH, 'usagers-2019.csv'), header=0, sep=";")

    # Chargement des fichiers lieux
    df_2019 = pd.read_csv(os.path.join(BASE_PATH, 'lieux-2019.csv'), sep=";")
    df_2020 = pd.read_csv(os.path.join(BASE_PATH, 'lieux-2020.csv'), sep=";")
    df_2021 = pd.read_csv(os.path.join(BASE_PATH, 'lieux-2021.csv'), sep=";")
    df_2022 = pd.read_csv(os.path.join(BASE_PATH, 'lieux-2022.csv'), sep=";")

    # Fusion des données lieux
    df_fuse = pd.concat([df_2019, df_2020, df_2021, df_2022], ignore_index=True)
    
    # Fusion des données usagers
    usagers_2019.drop_duplicates(inplace=True)
    usagers_2020.drop_duplicates(inplace=True)
    dfu_fuse = pd.concat([usagers_2022, usagers_2021, usagers_2020, usagers_2019], ignore_index=True)

    # Fusion finale avec la gravité
    df_fused_avec_grav = pd.merge(df_fuse, dfu_fuse[['Num_Acc', 'grav']], on='Num_Acc', how='left')
    df_fused_avec_grav['an'] = df_fused_avec_grav['Num_Acc'].astype(str).str[:4]

    # Fusion des caractéristiques
    fusion_carac = pd.concat([carac_2022, carac_2021, carac_2020, carac_2019], ignore_index=True)

    return df_fused_avec_grav, fusion_carac, dfu_fuse

try:
    df_fused_avec_grav, fusion_carac, fusion_usagers = load_data()
except Exception as e:
    st.error(f"Une erreur s'est produite lors du chargement des données : {str(e)}")
    st.stop()


name={"Accident_Id":"Num_Acc"}
carac_2022=carac_2022.rename(name,axis=1)
frames = [carac_2022, carac_2021, carac_2020,carac_2019]
fusion_carac = pd.concat(frames)

#Pour pouvoir utiliser les coordonnées GPS avec Geopandas je retraite les données et les convertis en float
fusion_carac["lat"]=fusion_carac["lat"].str.replace(",",".")
fusion_carac["lat"]=fusion_carac["lat"].astype("float")
fusion_carac["long"]=fusion_carac["long"].str.replace(",",".")
fusion_carac["long"]=fusion_carac["long"].astype("float")

#Pour pouvoir exploiter les données sur les départements je retraite les départements < à 10 en n°
fusion_carac.loc[(fusion_carac.dep == '1'), 'dep'] = '01'
fusion_carac.loc[(fusion_carac.dep == '2'), 'dep'] = '02'
fusion_carac.loc[(fusion_carac.dep == '3'), 'dep'] = '03'
fusion_carac.loc[(fusion_carac.dep == '4'), 'dep'] = '04'
fusion_carac.loc[(fusion_carac.dep == '5'), 'dep'] = '05'
fusion_carac.loc[(fusion_carac.dep == '6'), 'dep'] = '06'
fusion_carac.loc[(fusion_carac.dep == '7'), 'dep'] = '07'
fusion_carac.loc[(fusion_carac.dep == '8'), 'dep'] = '08'
fusion_carac.loc[(fusion_carac.dep == '9'), 'dep'] = '09'


name={"Accident_Id":"Num_Acc"}
carac_2022=carac_2022.rename(name,axis=1)
frames = [carac_2022, carac_2021, carac_2020,carac_2019]
fusion_carac = pd.concat(frames)

with open('/content/drive/MyDrive/Accidents de la route/Donnees/contour_departements.geojson',encoding='UTF-8') as dep:
    departement = geojson.load(dep)

st.title("Projet sur les Accidents de la route")
st.sidebar.title("Sommaire")
pages=["Introduction","Exploration", "Pré-processing", "Modélisation" ,"Déploiement du modèle","Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.write("### Projet réalisé par :")
st.sidebar.markdown("""- Nathalie MORVANT
- Naima TALAHIK
- Gabriel CIFCI
- Stéphane ROY""")


if page == pages[0] :
  st.write("# Introduction")



  st.header("Description du projets")
  st.write("""L’objectif de ce projet est d’essayer de prédire la gravité des accidents routiers en France.
         Les prédictions seront basées sur les données historiques. Nous avons fait le choix de nous baser sur 4 années de
         2019 à 2022.""")
  st.write("""Une première étape est d'explorer et de comprendre les jeux de données. Une deuxièmes étape est
         de rendre les jeux de données propres (valeurs manquantes, doublons, etc..) afin de les exploiter pour la mise en place
        des modèles de machine learning. """)

  st.header("Sources des jeux de données")
  st.write("""Notre source principale de données pour répondre à la problématique est le fichier national des accidents
           corporels de la circulation dit « Fichier BAAC » administré par l’Observatoire national interministériel de
           la sécurité routière "ONISR" et que l'on peut trouver sur data.gouv.""")
  st.write("""Les bases de données, extraites du fichier BAAC, répertorient l'intégralité des
           accidents corporels de la circulation, intervenus durant une année précise en France métropolitaine, dans les DOM
           et  TOM avec une description simplifiée.""")

  st.write("""Ils comprennent des informations de localisation de l’accident, telles que renseignées ainsi que des informations
           concernant les caractéristiques de l’accident et son lieu, les véhicules impliqués et leurs victimes.""")

  st.write("""Les fichiers sont disponibles en open data. Pour mener notre analyse nous avons récupéré les bases de données annuelles
           de 2019 à 2022 composées de 4 fichiers (Caractéristiques – Lieux – Véhicules – Usagers) au format csv.""")

  st.write("""Le n° d'identifiant de l’accident ("Num_Acc") présent dans les 4 fichiers permet d'établir un lien entre toutes
           les variables qui décrivent un accident. Quand un accident comporte plusieurs véhicules, le lien entre le véhicule
           et ses occupants est fait par la variable id_vehicule.""")
  st.link_button("Bases de données annuelles des accidents corporels de la circulation routière - Années de 2005 à 2022", "https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/")

  st.header("Autres sources de données")

  st.write("""Nous avons chercher à utiliser d'autres sources de données pour notre analyse. Ainsi nous avons chercher à mettre
           en rapport les accidents de la route avec le trafic routier quotidien par départements. Malheureusement seule l'année 2019 est disponible
           puisque le ministère des transport ne fournit plus ces données depuis. De plus nous nous sommes aperçus que le jeu de données
           fourni comportent des erreurs avec notamment un trafic routier très faible à Paris.""")

  st.write("""Nous avons également chercher à mettre en rapport le nombre d'habitants par départements avec les accidents de la route.
           Nous nous sommes procurés toutes ces données sur le site de data.gouv. et le site de l'INSEE.""")
  st.link_button("Trafic routier 2019", "https://www.data.gouv.fr/fr/datasets/trafic-moyen-journalier-annuel-sur-le-reseau-routier-national/")
  st.link_button("Nombre d'habitants par départements","https://www.insee.fr/fr/statistiques/1893198")









if page == pages[1] :
    st.write("### Exploration des jeux de données")
    st.header("1. Analyse du jeu de données Caractéristiques")
    st.write("Le jeu de données comporte 15 colonnes. En concaténant les 4 années nous obtenons un jeu de données de 218 404 observations")

    st.markdown("On affiche quelques graphiques pour explorer et visualiser ce jeu de données")


    nb_accident_collision=fusion_carac.loc[-(fusion_carac["col"]==-1)].reset_index()
    nb_accident_collision["col"]=nb_accident_collision["col"].map({1:"2 véhicules Frontale",2:"2 véhicules Arrière",3:"2 véhicules côté",4:"3 véhicules et + en chaîne",\
                                                 5:"3 véhicules et + col mulptiples", 6: "Autre collision",7:"Sans colision"})

    fig = px.histogram(nb_accident_collision, x="col", color='an',text_auto="count", labels={"an": "Années"})

    fig.update_layout(title = "Accidents par type de collisions et par années",
                  xaxis_title = 'Type de collisions',
                  yaxis_title = "Nombre d'accidents",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)


    nb_accident_meteo=fusion_carac.loc[-(fusion_carac["atm"]==-1)].reset_index()
    nb_accident_meteo["atm"]=nb_accident_collision["atm"].map({1:"Normale",2:"Pluie Légère",3:"Pluie forte",4:"Neige_grêle",\
                                                 5:"Brouillard", 6: "Vent fort",7:"Eblouissant",8:"Couvert",9:"Autre"})

    fig = px.histogram(nb_accident_meteo, x="atm", color='an',text_auto="count",labels={"an": "Années"})

    fig.update_layout(title = "Nombre d'accidents par type de conditions atmosphériques et par années",
                  xaxis_title = 'Conditions météos',
                  yaxis_title = "Nombre d'accidents",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)

    nb_accident_loca=fusion_carac.loc[-(fusion_carac["agg"]==-1)].reset_index()
    nb_accident_loca["agg"]=nb_accident_loca["agg"].map({1:"Hors Agglomération",2:"Agglomération"})

    fig = px.histogram(nb_accident_loca, x="agg", color='an',text_auto="count",labels={"an": "Années"})

    fig.update_layout(title = "Nombre d'accidents par localisation et par années",
                  xaxis_title = "Localisation",
                  yaxis_title = "Nombre",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)

    #on créé une colonne date en fusionnant l'année, le mois et le jour

    fusion_carac["date"]=fusion_carac.apply(lambda row: str(row["an"])+"-" + str(row["mois"])+"-" + str(row["jour"]) ,axis=1)
    # L'argument 'yearfirst=True' est utilisé pour indiquer que le format de la date est "AAAA-MM-JJ"
    fusion_carac["date"]=pd.to_datetime(fusion_carac["date"],yearfirst = True)

    #on compte le nombre d'accidents par date
    values=fusion_carac["date"].value_counts().sort_index().reset_index()
    nom={"count":"Nombre"}
    values=values.rename(nom,axis=1)

    fig = px.line(values,x = "date",y="Nombre")

    fig.add_scatter(x=values["date"], y=[np.median(values["Nombre"])]*len(values["date"]),mode='lines',name="Médiane",marker_color='rgba(255, 182, 193, .9)')


    fig.update_layout(title = "Nombre d'accidents par jours",
                  xaxis_title = "Nombre d'accidents par jour",
                  yaxis_title = "Nombre",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)

    fig = px.histogram(nb_accident_loca, x="mois", color='an',text_auto="count",labels={"an": "Années"})

    fig.update_layout(title = "Nombre d'accidents par mois et par années",
                  xaxis_title = "Mois",
                  yaxis_title = "Nombre",
                  barmode='group',
                  xaxis = dict(
                  tickvals = [1,2,3,4,5,6,7,8,9,10,11,12],
                  ticktext  = ["janvier","février","mars","avril","mai","juin","juillet","août","septembre","octobre","novembre","décembre"]))
    fig.update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(fig)

    #on compte le nombre d'accident par jour de la  semaine et par années
    values_par_jour=fusion_carac.groupby([fusion_carac["date"].dt.weekday,fusion_carac["an"]]).agg({"Num_Acc":"count"}).reset_index()
    nom={"Num_Acc":"Nombre"}
    values_par_jour=values_par_jour.rename(nom,axis=1)
    values_par_jour["date"]=values_par_jour["date"].map({0:"Lundi",1:"Mardi",2:"Mercredi",3:"Jeudi",4:"Vendredi",5:"Samedi",6:"Dimanche"})

    fig = px.histogram(values_par_jour, x="date",y="Nombre", color='an',text_auto="count",labels={"an": "Années"})

    fig.add_scatter(x=values_par_jour["date"], y=[np.median(values_par_jour["Nombre"])]*len(values_par_jour["date"]),mode='lines',name="Médiane")

    fig.update_layout(title = "Nombre d'accidents par jour dans la semaine et par années",
                  xaxis_title = "jour",
                  yaxis_title = "Nombre",
                  barmode='group')
    st.plotly_chart(fig)

    fusion_carac['heure'] = fusion_carac['hrmn'].apply(lambda x : x[0:2]).astype("int")
    fig = px.histogram(fusion_carac, x="heure", color='an',text_auto="count",labels={"an": "Années"})


    fig.update_layout(title = "Nombre d'accidents par heure dans la journée et par années",
                  xaxis_title = "Heures dans la journée",
                  yaxis_title = "Nombre",
                  barmode='group')
    st.plotly_chart(fig)

    accident_2022=carac_2022.groupby("dep").agg({"Num_Acc":"count"}).reset_index().sort_values(by="Num_Acc",ascending=False)
    accident_2022["année"]=2022
    accident_2021=carac_2021.groupby("dep").agg({"Num_Acc":"count"}).reset_index().sort_values(by="Num_Acc",ascending=False)
    accident_2021["année"]=2021
    accident_2020=carac_2020.groupby("dep").agg({"Num_Acc":"count"}).reset_index().sort_values(by="Num_Acc",ascending=False)
    accident_2020["année"]=2020
    accident_2019=carac_2019.groupby("dep").agg({"Num_Acc":"count"}).reset_index().sort_values(by="Num_Acc",ascending=False)
    accident_2019["année"]=2019

    liste_accident=[accident_2019,accident_2020,accident_2021,accident_2022]
    accident_fusion=pd.concat(liste_accident,axis=0).reset_index().drop(columns="index")

    fig = px.box(accident_fusion,y="Num_Acc", x='année',hover_data=["dep"])

    fig.update_layout(title = "Distribution du nombre d'accidents par années selon le département",
                  xaxis_title = "Années",
                  yaxis_title = "Nombre d'accidents")
    st.plotly_chart(fig)

    fusion_carac_usagers=fusion_carac.merge(right=fusion_usagers,on="Num_Acc",how="left")
    fusion_carac_usagers["agg"]=fusion_carac_usagers["agg"].map({1:"Hors Agglomération",2:"Agglomération"})

    fusion_carac_usagers["grav"]=fusion_carac_usagers["grav"].map({1:"Indemne",2:"Tué",3:"Bléssé hospitalisé",4:"Bléssé léger"})

    fig = px.histogram(fusion_carac_usagers, x="grav", color='agg',text_auto="count",animation_frame="an",labels={"agg": "Localisation"})

    fig.update_layout(title = "Accidents par gravité et en fonction de la localisation",
                  xaxis_title = "Gravité",
                  yaxis_title = "Nombre d'accidents",
                  barmode='group')
    fig.update_xaxes(categoryorder = 'total descending')

    st.plotly_chart(fig)

 #on importe le fichier json permettant les contours des départements



    for feature in departement['features']:
        feature['id']= feature['properties']['code']

       #avec ce bout de code on récupère dans un dictionnaire le Numéro du département et son nom.
    dico_dep={}
    for feature in departement['features']:
        dico_dep[feature['properties']['code']]=feature['properties']['nom']

    df_département = pd.DataFrame(dico_dep.items(), columns=['Département', 'Nom'])

    liste_outre_mer=["971","972","973","974",'977', '978', '975',"976","987","988","986"]
    fusion_carac_dep=fusion_carac[~fusion_carac["dep"].isin(liste_outre_mer)]

    dep_2022=fusion_carac_dep.loc[fusion_carac_dep["an"]==2022]["dep"].value_counts().to_frame().reset_index()
    dep_2022["echelle_accident"]=np.log10(dep_2022['count'])
    dep_2022["an"]=2022


    dep_2021=fusion_carac_dep.loc[fusion_carac_dep["an"]==2021]["dep"].value_counts().to_frame().reset_index()
    dep_2021["echelle_accident"]=np.log10(dep_2021['count'])
    dep_2021["an"]=2021


    dep_2020=fusion_carac_dep.loc[fusion_carac_dep["an"]==2020]["dep"].value_counts().to_frame().reset_index()
    dep_2020["echelle_accident"]=np.log10(dep_2020['count'])
    dep_2020["an"]=2020


    dep_2019=fusion_carac_dep.loc[fusion_carac_dep["an"]==2019]["dep"].value_counts().to_frame().reset_index()
    dep_2019["echelle_accident"]=np.log10(dep_2019['count'])
    dep_2019["an"]=2019



    dep_tous=pd.concat((dep_2022,dep_2021,dep_2020,dep_2019),axis=0)


    dico={"count":"Nombre_accidents","dep":"Département", "an":"Année"}
    dep_tous=dep_tous.rename(dico,axis=1)
    dep_tous["Département"]=dep_tous["Département"].astype("str")
    dep_tous=dep_tous.merge(right=df_département,on="Département",how="left")
    dep_tous["Num_Nom"]=dep_tous["Département"]+ " "+ dep_tous["Nom"]

    fig3 = px.choropleth_mapbox(dep_tous, locations = 'Département',
                            geojson= departement,
                            color='echelle_accident',
                            color_continuous_scale=["green","orange","red"],
                            range_color=[2,3.5],
                            hover_name="Num_Nom",
                            animation_frame='Année',
                            hover_data=['Nombre_accidents'],
                            title="Carte de répartition des accidents en France et par années",
                            mapbox_style="open-street-map",
                            center= {'lat':46, 'lon':2},
                            zoom =4,
                            opacity= 0.6, width=2000, height=2000)

    st.plotly_chart(fig3)

st.header("2. Analyse du jeu de données Lieux")


# Définition des mappings
catr_mapping = {
    1: 'Autoroute', 2: 'Route nationale', 3: 'Route Départementale',
    4: 'Voie Communales', 5: 'Hors réseau public',
    6: 'Parc de stationnement ouvert à la circulation publique',
    7: 'Routes de métropole urbaine', 9: 'autre'
}

circ_mapping = {
    -1: 'Non renseigné', 1: 'A sens unique', 2: 'Bidirectionnelle',
    3: 'A chaussées séparées', 4: 'Avec voies d affectation variable',
}

vosp_mapping = {
    -1: 'Non renseigné', 0: 'Sans objet', 1: 'Piste cyclable',
    2: 'Bande cyclable', 3: 'Voie réservée',
}

prof_mapping = {
    -1: 'Non renseigné', 1: 'Plat', 2: 'Pente',
    3: 'Sommet de côte', 4: 'Bas de côte',
}

plan_mapping = {
    -1: 'Non renseigné', 1: 'Partie rectiligne', 2: 'En courbe à gauche',
    3: 'En courbe à droite', 4: 'En S',
}

surf_mapping = {
    -1: 'Non renseigné', 1: 'Normal', 2: 'Mouillée', 3: 'Flaques',
    4: 'Inondée', 5: 'Enneigée', 6: 'Boue', 7: 'Verglacée',
    8: 'Corps gras – huile', 9: 'Autre'
}

infra_mapping = {
    -1: 'Non renseigné', 0: 'Aucun', 1: 'Souterrain-tunnel', 2: 'Pont',
    3: 'Bretelle d échangeur ou de raccordement', 4: 'Voie ferrée',
    5: 'Carrefour aménagé', 6: 'Zone piétonne', 7: 'Zone de péage',
    8: 'Chantier', 9: 'Autre'
}

grav_mapping = {
    1: 'Indemne', 2: 'Tué', 3: 'Blessé hospitalisé', 4: 'Blessé léger'
}



# Visualisation pour catr
st.header("Catégorie de route")
catr_an_compte = df_fused_avec_grav.groupby(['catr', 'an']).size().reset_index(name='Nombre accidents')
catr_an_compte['catr'] = catr_an_compte['catr'].map(catr_mapping)
fig_catr = px.bar(catr_an_compte, x='catr', y='Nombre accidents', color='an', barmode='group', 
                  title='Nombre d\'accidents par catégorie de route et par an')
st.plotly_chart(fig_catr)

# Visualisation pour circ
st.header("Régime de circulation")
circ_an_compte = df_fused_avec_grav.groupby(['circ', 'an']).size().reset_index(name='Nombre accidents_circ')
circ_an_compte['circ'] = circ_an_compte['circ'].map(circ_mapping)
fig_circ = px.bar(circ_an_compte, x='circ', y='Nombre accidents_circ', color='an', barmode='group', 
                  title='Nombre d\'accidents par régime de circulation et par an')
st.plotly_chart(fig_circ)

# Visualisation pour vosp
st.header("Voie spéciale")
vosp_an_compte = df_fused_avec_grav.groupby(['vosp', 'an']).size().reset_index(name='Nombre accidents_vosp')
vosp_an_compte['vosp'] = vosp_an_compte['vosp'].map(vosp_mapping)
fig_vosp = px.bar(vosp_an_compte, x='vosp', y='Nombre accidents_vosp', color='an', barmode='group', 
                  title='Nombre d\'accidents par signal voie reservée')
st.plotly_chart(fig_vosp)

# Visualisation pour prof
st.header("Profil de la route")
prof_an_compte = df_fused_avec_grav.groupby(['prof', 'an']).size().reset_index(name='Nombre accidents_prof')
prof_an_compte['prof'] = prof_an_compte['prof'].map(prof_mapping)
fig_prof = px.bar(prof_an_compte, x='prof', y='Nombre accidents_prof', color='an', barmode='group', 
                  title='Nombre d\'accidents par dénivelé')
st.plotly_chart(fig_prof)

# Visualisation pour plan
st.header("Tracé en plan")
plan_an_compte = df_fused_avec_grav.groupby(['plan', 'an']).size().reset_index(name='Nombre accidents_plan')
plan_an_compte['plan'] = plan_an_compte['plan'].map(plan_mapping)
fig_plan = px.bar(plan_an_compte, x='plan', y='Nombre accidents_plan', color='an', barmode='group', 
                  title='Nombre d\'accidents par plan et par an')
st.plotly_chart(fig_plan)

# Visualisation pour surf
st.header("État de la surface")
surf_an_compte = df_fused_avec_grav.groupby(['surf', 'an']).size().reset_index(name='Nombre accidents_surf')
surf_an_compte['surf'] = surf_an_compte['surf'].map(surf_mapping)
fig_surf = px.bar(surf_an_compte, x='surf', y='Nombre accidents_surf', color='an', barmode='group', 
                  title='Nombre d\'accidents par état de surface et par an')
st.plotly_chart(fig_surf)

# Visualisation pour infra
st.header("Infrastructure")
infra_an_compte = df_fused_avec_grav.groupby(['infra', 'an']).size().reset_index(name='Nombre accidents_infra')
infra_an_compte['infra'] = infra_an_compte['infra'].map(infra_mapping)
fig_infra = px.bar(infra_an_compte, x='infra', y='Nombre accidents_infra', color='an', barmode='group', 
                   title='Nombre d\'accidents par infrastructure et par an')
st.plotly_chart(fig_infra)

# Visualisation pour vma
st.header("Vitesse maximale autorisée")
df_fused_avec_grav['grav_mapped'] = df_fused_avec_grav['grav'].map(grav_mapping)
vma_an_grav_compte = df_fused_avec_grav.groupby(['vma', 'an', 'grav_mapped']).size().reset_index(name='Nombre accidents')
pivot_table = vma_an_grav_compte.pivot_table(index=['vma', 'an'], columns='grav_mapped', values='Nombre accidents', fill_value=0)
pivot_table_mean = pivot_table.groupby('vma').mean()

scaler = MinMaxScaler()
pivot_table_mean_normalized = pd.DataFrame(scaler.fit_transform(pivot_table_mean), 
                                           columns=pivot_table_mean.columns, 
                                           index=pivot_table_mean.index)

fig_vma = px.line(pivot_table_mean_normalized.reset_index(), x='vma', y=pivot_table_mean_normalized.columns,
                  title='Nombre moyen d\'accidents par VMA et par catégorie de gravité (Normalisé)')
fig_vma.update_layout(xaxis_title='VMA', yaxis_title='Nombre moyen d\'accidents (Normalisé)',
                      legend_title='Gravité')
fig_vma.update_xaxes(range=[0, 200])
st.plotly_chart(fig_vma)


if page == pages[2]:
    st.write("# Preprocessing et structuration du dataset")

    # Nombre de véhicules par accident
    temp = fusion_usagers.loc[:, ['Num_Acc', 'id_vehicule']]
    temp = temp.drop_duplicates(keep='first')
    count_vehicule = temp.groupby('Num_Acc').agg({'id_vehicule': 'count'}).reset_index()
    count_vehicule = count_vehicule.rename(columns={'id_vehicule': 'nbr_vehicule'})

    # Nombre d'usagers par accident
    count_usager = fusion_usagers["Num_Acc"].value_counts().sort_index().reset_index()
    count_usager = count_usager.rename(columns={'count': 'nbr_usager_acc'})

    # Présence de piétons
    temp = fusion_usagers.loc[:, ['Num_Acc', 'catu']]
    temp['pieton'] = np.where(temp['catu'] == 3, 1, 0)
    pieton_groupby = temp.groupby('Num_Acc').agg({'pieton': 'sum'})
    pieton_groupby['avec_pieton'] = np.where(pieton_groupby['pieton'] > 0, 1, 0)

    # Gravité des accidents
    temp = fusion_usagers.loc[:, ['Num_Acc', 'grav']]
    temp['tue'] = np.where(temp['grav'] == 2, 1, 0)
    temp['blesse_leger'] = np.where(temp['grav'] == 4, 1, 0)
    temp['blesse_hospitalise'] = np.where(temp['grav'] == 3, 1, 0)
    grav_groupby = temp.groupby('Num_Acc').agg({'tue': 'sum', 'blesse_leger': 'sum', 'blesse_hospitalise': 'sum'})

    conditionlist = [
        (grav_groupby['tue'] > 0),
        (grav_groupby['blesse_hospitalise'] > 0) & (grav_groupby['tue'] == 0),
        (grav_groupby['blesse_leger'] > 0) & (grav_groupby['blesse_hospitalise'] == 0) & (grav_groupby['tue'] == 0)
    ]
    choicelist = ["mortel", "grave", "leger"]
    grav_groupby["grav_Acc"] = np.select(conditionlist, choicelist, default="NR")

    # Interface utilisateur
    st.header("1. Traitement des données véhicules et usagers")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution du nombre de véhicules par accident")
        fig_vehicles = px.histogram(count_vehicule, x="nbr_vehicule",
                                    title="Distribution du nombre de véhicules par accident")
        st.plotly_chart(fig_vehicles, use_container_width=True)

    with col2:
        st.subheader("Distribution du nombre d'usagers par accident")
        fig_users = px.histogram(count_usager, x="nbr_usager_acc",
                                 title="Distribution du nombre d'usagers par accident")
        st.plotly_chart(fig_users, use_container_width=True)

    st.header("2. Analyse de la présence de piétons")
    fig_pedestrians = px.pie(pieton_groupby, names="avec_pieton",
                             title="Proportion d'accidents avec piétons",
                             values="avec_pieton")
    st.plotly_chart(fig_pedestrians, use_container_width=True)

    st.header("3. Catégorisation de la gravité des accidents")
    grav_counts = grav_groupby["grav_Acc"].value_counts().reset_index()
    grav_counts.columns = ['gravite', 'count']
    fig_severity = px.bar(grav_counts, x="gravite", y="count",
                          title="Distribution de la gravité des accidents")
    st.plotly_chart(fig_severity, use_container_width=True)



    import streamlit as st
    import pandas as pd

    # Chemin d'accès aux fichiers
    file_paths = {
        'Caractéristiques 2022': '/content/drive/MyDrive/Accidents de la route/Donnees/caracteristiques-2022.csv',
        'Caractéristiques 2021': '/content/drive/MyDrive/Accidents de la route/Donnees/caracteristiques-2021.csv',
        'Caractéristiques 2020': '/content/drive/MyDrive/Accidents de la route/Donnees/caracteristiques-2020.csv',
        'Caractéristiques 2019': '/content/drive/MyDrive/Accidents de la route/Donnees/caracteristiques-2019.csv',
        'Usagers 2022': '/content/drive/MyDrive/Accidents de la route/Donnees/usagers-2022.csv',
        'Usagers 2021': '/content/drive/MyDrive/Accidents de la route/Donnees/usagers-2021.csv',
        'Usagers 2020': '/content/drive/MyDrive/Accidents de la route/Donnees/usagers-2020.csv',
        'Usagers 2019': '/content/drive/MyDrive/Accidents de la route/Donnees/usagers-2019.csv'
}

# Lire et afficher les noms des colonnes pour chaque fichier
    for description, path in file_paths.items():
          data = pd.read_csv(path, header=0, sep=';')
          st.write(f"Colonnes pour {description}:")
          st.write(data.columns.tolist())


    st.header("4. Recodage des variables")

    st.subheader("Catégorisation des véhicules")

    @st.cache
    def categorize_vehicle(catv):
        if pd.isna(catv) or catv == 0:
            return "Engin spécial"
        elif catv in [2, 4, 5, 6, 30, 31, 32, 33, 34, 41, 42, 43]:
            return "Deux Roues"
        elif catv in [3, 7, 8, 9, 10, 11, 12]:
            return "Véhicule léger"
        elif catv in [13, 14, 15, 16, 17]:
            return "Poids lourd"
        elif catv in [18, 19, 37, 38, 39, 40]:
            return "Transport en commun"
        elif catv == 1:
            return "Vélo"
        else:
            return "Engin spécial"

    fusion_usagers['categorie_vehicule'] = fusion_usagers['catv'].apply(categorize_vehicle)
    vehicle_cat_counts = fusion_usagers['categorie_vehicule'].value_counts()
    fig_vehicle_cat = px.pie(values=vehicle_cat_counts.values, names=vehicle_cat_counts.index,
                            title="Répartition des catégories de véhicules impliqués dans les accidents")
    st.plotly_chart(fig_vehicle_cat, use_container_width=True)


    st.subheader("Simplification des conditions météorologiques")
    fusion_carac['meteo_simple'] = fusion_carac["atm"].map({
        1: "Normales",
        2: "Anormales", 3: "Anormales", 4: "Anormales", 5: "Anormales",
        6: "Anormales", 7: "Anormales", 8: "Anormales", 9: "Anormales"
    })
    meteo_counts = fusion_carac['meteo_simple'].value_counts()
    fig_meteo = px.pie(values=meteo_counts.values, names=meteo_counts.index,
                       title="Répartition des conditions météorologiques lors des accidents")
    st.plotly_chart(fig_meteo, use_container_width=True)

    st.header("5. Traitement des variables temporelles")
    fusion_carac['heure'] = fusion_carac['hrmn'].apply(lambda x: int(x[:2]))
    fusion_carac['moment_journee'] = pd.cut(fusion_carac['heure'],
                                            bins=[-1, 6, 12, 18, 24],
                                            labels=['Nuit', 'Matin', 'Après-midi', 'Soir'])
    moment_counts = fusion_carac['moment_journee'].value_counts()
    fig_moment = px.pie(values=moment_counts.values, names=moment_counts.index,
                        title="Répartition des accidents selon le moment de la journée")
    st.plotly_chart(fig_moment, use_container_width=True)

    st.write("""
    Ces opérations de prétraitement nous ont permis de structurer et de nettoyer les données,
    réduisant la complexité du jeu de données tout en préservant les informations essentielles
    pour l'analyse et la modélisation ultérieures.
    """)



if page == pages[3] :
    st.write("### Méthodologie de modélisation + performance")

if page == pages[4] :
        st.write("### Déploiemement du modèle")


if page == pages[5] :
        st.write("### Conclusion (retours critiques, pistes améliorations ...)")