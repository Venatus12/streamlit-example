from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np

st.markdown('# Introductie')
st.markdown('Voor de dashboard-opdracht hebben wij gekozen voor het bestuderen van Netflix films. '
         'Met zo’n 220 miljoen abonnees, 3.744 films, en zo’n 70 originele films, '
         'is Netflix de meest populaire video streaming dienst ter wereld. Via Kaggle hebben wij de meest recente Netflix films & series dataset verkregen. '
         'Deze dataset bevat onder meer: de titel, jaar van uitgave, tijdsduur, genres keywords, imbd score, imbd votes, tmdb populariteit en score van '
         'series en films tussen 1954 em 2022. Ook hebben we via Kaggle een dataset verkregen waar alle films in The Movie Database (TMDb) zijn verzameld.'
         'Deze dataset bevat: titel, jaar van uitgave, tmdb populariteit, buget, winst, en nog eens 14 andere factoren.')

st.markdown('Wij hebben gekeken naar:'
         '- Het verschil in tijdsduur tussen genres'
         '- Welke genre is het meest populaire op Netflix'
         '- In welk jaar zijn de meeste films uitgekomen'
         '- Hoe populaire zijn de nieuwste film'
         '- Het budget, de omzet en de winst van Netflix films')

st.markdown('# Data manipulatie')

# COR OMZET
st.markdown('## Omzetten van IMDb ID')
st.write("Omdat wij graag willen kijken naar de bugget en winst van elke Netflix film hebben wij een andere dataset nodig. "
         "De Netflix-dataset bevat immers niet de winsten en buggeten van films. Een database die deze twee wel bevat is de "
         "The Movie Database (TMDb). Elke film in deze dataset bevat een unieke TMDb ID nummer die we kunnen koppelen aan de Netflix-dataset. "
         "Echter is er een kwestie: de Netflix dataset bevat geen TMDb ID, maar een IMDb ID. Door middel van de TMDb API kunnen we de IMDb ID "
         "omzetten naar de TMDb ID aan de hand van de /find methode van de TMDb API. De functies hieronder zet de IMDb ID's in de Netflix dataset onm in TMDb ID's. "
         "Omdat een aantal Netflixfilms geen IMDb ID bevatten, hebben we de TMDb ID's verkregen aan de hand van de title en jaar van uitgaven van elke film. "
         "Bij Netflixfilms waar geen TMDb ID verkregen kon worden, werd een 'NO_MATCH' string gegeven.")

transfor1_code = """
def imdbid_to_tmdbid(row):
    api_key = '4412429954772dcd3e31d27135911113'

    # Alle films (row) waar de imdb_id is aanwezig wordt de title en id geisoleerd.
    if row['imdb_id'] != None and pd.notna(row['imdb_id']):
        imdb_id = row['imdb_id']
        title = row['title']

        # Vervolgens wordt de TMDb ID via de TMDb-API opgevraagd aan de hand van de imdb_id.
        api_request = f'https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id'
        data = requests.get(url=api_request).json()['movie_results']

        # TMDb ID en title wordt verkregen uit de data.json. Wanneer de filmtitle uit de data.json
        # overeenkomt met de gegeven filmtitle wordt de TMDb ID toegewezen aan de film.
        if data:
            tmdb_id = data[0]['id']
            tmdb_title = data[0]['title']
            if title == tmdb_title:
                row['tmdb_id'] = tmdb_id

    # De aangepaste film met de nieuwe TMDb ID wordt teruggegeven.
    return row
"""

transfor2_code = """
def id_by_title(row):
    api_key = '4412429954772dcd3e31d27135911113'

    # Alle films (row) waar de TMDb ID niet is verkregen via de IMDb ID.
    if pd.isna(row['tmdb_id']):
        title = row['title']
        year = row['release_year']

        # Vervolgens wordt de TMDb ID via de TMDb-API opgevraagd aan de hand van de film title en jaar van uitgave.
        api_request = f'http://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}&year={year}'
        data = requests.get(url=api_request).json()

        # Wanneer de TMDb ID nogmaals niet kan worden verkregen, betekend het dat er geen match is tussen de TMDb en de Netflix film.
        if len(data['results']) == 0:
            row['tmdb_id'] = 'NO_MATCH'
            return row

        if len(data['results']) >= 0:
            id = data['results'][0]['id']
            row['tmdb_id'] = id

    return row
"""

st.code(transfor1_code, language='python')
st.code(transfor2_code, language='python')

# JADE MERGE
st.markdown('## Samenvoegen van datasets')
st.write('Met behulp van de onderstaande code worden de budget en revenue uit de TMDb dataset samengevoegd met de Netflix dataset. '
         'Allereerst wordt TMDb dataset gefilterd op "budget", "revenue" en "id". Vervolgens wordt er met behulp van de "id", die '
         'zich in beide datasets bevindt, de datasets samengevoegd. Vervolgens wordt er een nieuwe kolom "winst" gemaakt door het budget '
         'van de omzet af te trekken. Tot slot verwijderen we alle NaN waardes die de waarde 0 hebben gekregen, omdat deze ons niks kunnen vertellen over de data.')

merge_code = '''
# Verander de waardes van de kolom tmdb_id van een float naar een integer
netflix['tmdb_id'] = netflix['tmdb_id'].str.replace(r'NO_MATCH', '0') # Verander 'NO_MATCH' naar het getal '0'
netflix.tmdb_id = netflix.tmdb_id.astype('float').astype('Int64') # Verander alle waardes eerst naar float en dan naar integer

# Maak nieuwe dataframe met alleen id, budget en revevue kolommen
movies_id = movies.loc[:, ['id', 'budget', 'revenue']]

# Geef de kolom id de naam tmdb_id zoals in de dataframe 'netflix'
movies_id.rename(columns = {'id':'tmdb_id'}, inplace = True)

# Voeg de kolommen 'budget' en 'revenue' toe aan de dataframe 'netflix', hier maken we een nieuwe dataframe van genaamd: df
df = pd.merge(netflix, movies_id, on = 'tmdb_id')

# Voeg nieuwe kolom toe aan 'df' met de winst door de 'revenue' min het 'budget' te doen
df['winst'] = df.revenue - df.budget

# Geef alle NaN waardes de waarde 0 om makkelijker te kunnen plotten met de dataset
df['imdb_score'] = df['imdb_score'].fillna(0)

# Verwijder alle waardes met nul omdat deze ons niks vertellen en deze willen we dus ook niet in onze plots zien
df = df[df.imdb_score != 0.0]
df = df1[df1.winst != 0.0]
df = df1[df1.revenue != 0.0]
df = df1[df1.budget != 0.0]
'''

st.code(merge_code, language='python')
