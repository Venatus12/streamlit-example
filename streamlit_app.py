from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from pillow import Image


netflix = pd.read_csv('netflixwinst.csv')
genre_data = pd.read_csv('genre_data.csv', index_col=0)

st.markdown('# Introductie')
st.markdown('Voor de dashboard-opdracht hebben wij gekozen voor het bestuderen van Netflix films. '
         'Met zo’n 220 miljoen abonnees, 3.744 films, en zo’n 70 originele films, '
         'is Netflix de meest populaire video streaming dienst ter wereld. Via Kaggle hebben wij de meest recente Netflix films & series dataset verkregen. '
         'Deze dataset bevat onder meer: de titel, jaar van uitgave, tijdsduur, genres keywords, imbd score, imbd votes, tmdb populariteit en score van '
         'series en films tussen 1954 em 2022. Ook hebben we via Kaggle een dataset verkregen waar alle films in The Movie Database (TMDb) zijn verzameld.'
         'Deze dataset bevat: titel, jaar van uitgave, tmdb populariteit, buget, winst, en nog eens 14 andere factoren.')

st.markdown("""
Wij hebben gekeken naar:
- Het verschil in tijdsduur tussen genres
- Welke genre is het meest populaire op Netflix
- In welk jaar zijn de meeste films uitgekomen
- Hoe populaire zijn de nieuwste film'
- Het budget, de omzet en de winst van Netflix films
""")


st.markdown('## Data importeren')
st.write('Voor het verkrijgen van de dataset hebben we onderstaande code gebruikt. Om de Kaggle API te gebruiken moet er een kaggle-api-key worden aangemaakt. Dit kan alleen als je een account op Kaggle hebt. Vervolgens kan je de dataset naar keuze ophalen en downloaden van het web.')
api_code ="""
import os
os.environ['KAGGLE_USERNAME'] = "<your-kaggle-username>"
os.environ['KAGGLE_KEY'] = "<your-kaggle-api-key>"

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_download_file('victorsoeiro/netflix-tv-shows-and-movie', file_name='titles.csv') # Netflix movies/series
api.dataset_download_file('akshaypawar7/millions-of-movies', file_name='movies.csv') # TMDb movies

# # https://towardsdatascience.com/downloading-datasets-from-kaggle-for-your-ml-project-b9120d405ea4
"""



st.markdown('# Data manipulatie')

# COR OMZET
st.markdown('**Omzetten van IMDb ID**')
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
st.markdown('**Samenvoegen van datasets**')
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

st.markdown('# Introductie')
st.markdown("""## Budget, de omzet en de winst van Netflix films""")

st.write('Het onderstaande figuur laat de relatie zien tussen de Winst en de Imdb score. Het valt op dat de twee variabelen weinig samenhang hebben. De film met de laagste imdb score is de film "Me Against You: Mr. Ss Vendetta" maar deze film heeft wel een kleine winst gemaakt. Verder valt Titanic natuurlijk weer op met een hele hoge winst.')

fig = px.scatter(netflix, x = 'imdb_score', y = 'winst', title = 'Boxplot van de relatie tussen de imdb score en de winst',
                 color = 'title',
                 labels = {'imdb_score' : 'Imdb score',
                           'winst' : 'Winst',
                           'title':'Titel'},
                 log_y = True,
                 template='none')
fig.update_layout(showlegend = False, hovermode="closest")
fig.show()

scatter_imdb_winst = fig
st.plotly_chart(scatter_imdb_winst, use_container_widtdh=True)

st.write('In het onderstaande figuur is de relatie tussen de imdb score en de omzet te zien. Let op dat de y-as op een logaritmische schaal is. Er is te zien dat de film "Raja Natwarlal" een prima imdb score van 6.1 heeft gehaald maar slechts een omzet van 4 heeft gemaakt. Verder heeft de Titanic weer de hoogste omzet en ook goede imdb score van 7.9. Er is een lichte samenhang te zien tussen de twee variabelen.')

fig = px.scatter(netflix, x = 'imdb_score', y = 'revenue', title = 'Boxplot van de relatie tussen de imdb score en de omzet',
                 color = 'title',
                 labels = {'imdb_score': 'Imdb score',
                           'revenue': 'Omzet',
                           'title':'Titel'},
                 log_y = True,
                 template='none')
fig.update_layout(showlegend = False, hovermode="closest")
fig.show()

scatter_imdb_omzet = fig
st.plotly_chart(scatter_imdb_omzet, use_container_widtdh=True)

st.write('In de onderstaande figuur is de relatie tussen de imdb score en het budget te zien. Er is te zien dat de film met het hoogste budget de film "The Dark Knight Rises" is. Deze film is ook behoorlijk hoog beoordeeld met een imdb score van 8.4. Er is verder geen duidelijk samenhang te zien tussen de twee variabelen. Wel is het zo dat bijna alle laag beoordeelde films ook een laag budget hebben. De film "The Cat in the Hat" is hiervan een uitzondering. Deze film had een hoog budget maar een lage imdb score.')

fig = px.scatter(netflix, x = 'imdb_score', y = 'budget', title = 'Boxplot van de relatie tussen de imdb score en het budget',
                 color = 'title',
                 labels = {'imdb_score': 'Imdb score',
                           'budget': 'Budget',
                           'title':'Titel'},
                 log_y = False, template='none')
fig.update_layout(showlegend = False, hovermode="closest")
fig.show()
scatter_imdb_budget = fig
st.plotly_chart(scatter_imdb_budget, use_container_widtdh=True)

st.write('In de onderstaande figuur is de relatie tussen de omzet en het budget te zien. Er is een lineaire samenhang te zien tussen deze twee variabelen. Over het algemeen geldt dat hoe hoger het budget, hoe hoger de omzet. Er zijn wel een aantal uitschieters. De meest op vallende is de "Titanic" met een veel hogere omzet. Daarnaast vallen "Red Notice" en "The Irishman" ook op met beide een hoog budget maar lage omzet.')

fig = px.scatter(netflix, x = 'budget', y = 'revenue', title = 'Boxplot van de relatie tussen de omzet en het budget',
                 color = 'title',
                 labels = {'revenue': 'Omzet',
                           'budget': 'Budget',
                           'title':'Titel'},
                 log_y = False, template='none')
fig.update_layout(showlegend = False, hovermode="closest")
fig.show()
scatter_budget_revenue = fig
st.plotly_chart(scatter_budget_revenue, use_container_widtdh=True)

st.write('In de onderstaande figuur is de relatie tussen het jaar van uitgave en de winst per titel te zien. Er is te zien dat in de jaren 1997, 2011, 2012 en 2017 de meeste succesvolle films zijn uitgebracht op het gebied van winst. Het jaar 1997 heeft dit vooral te danken aan de "Titanic". Het jaar 2012 heeft populaire films als "The Dark Knight Rises" en "The Amazing Spider-Man". Opvallend is dat tot het jaar 1993 heel weinig winst is gemaakt in de filmindustrie. Hetzelfde geldt voor de jaren 2020 en 2022. Ook is het zo dat na het jaar 2012 er seeds meer verlies wordt gemaakt in de film industrie.')

fig = px.bar(netflix, x = 'release_year', y = 'winst', title = 'Barplot van de relatie tussen het jaar van uitgave en de winst',
             color = 'title',
             labels = {'release_year':'Released',
                       'winst':'Winst',
                       'title':'Titel'},
             log_y = False, template='none')

fig.update_layout(showlegend=False, hovermode="closest")
fig.show()

bar_jaar_winst = fig
st.plotly_chart(bar_jaar_winst, use_container_widtdh=True)

st.markdown("""## Tijdsduur, populariteit en score van Netflix films per genre""")
st.write('In onderstaand figuur zijn vier histogrammen afgebeeld met daarin de tijdsduur, populariteit en de scores van Netflix films. De tijdsduur van een gemiddelde Netflix film is zo’n 90 tot 120 minuten. We zien ook dat er een significante hoeveelheid films met een tijdsduur van 30 tot 40 min en 60 min aanwezig zijn. Films van meer dan 150 min komen relatief weinig voor.')

st.write('Wanneer we naar de populariteit van alle Netflix films kijken, is te zien dat de meeste Netflix films een TMDb populariteit van minder dan 1000 hebben. De gemiddelde score van Netflix films ligt tussen de 6 en 7.')

# sns.set_theme(style='white')
# fig, axes = plt.subplots(2, 2, figsize=(18,12), dpi=80)
# sns.set(font_scale = 1.2)
# sns.histplot(data=netflix, x='runtime', ax=axes[0][0], kde=True, element='step', color='b').set(title='Tijdsduur van films in min', xlabel=None)
# sns.histplot(data=netflix, x='imdb_score', ax=axes[1][0], kde=True, element='step', color='r').set(title='IMDb score', xlabel=None)
# sns.histplot(data=netflix, x='tmdb_score', ax=axes[1][1], kde=True, element='step', color='g').set(title='TMDb score', xlabel=None)
# sns.histplot(data=netflix, x='tmdb_popularity', ax=axes[0][1], kde=True, element='step', log_scale=True, color='gray').set(title='TMDb popularity', xlabel=None)
# hist_combined = fig
# plt.show()

hist_combined = Image.open('hist_combined.png')
st.image(hist_combined)


def data_per_genre_dict(dataset, genre_list):
    data_dict = {}

    for index, row in dataset.iterrows():

        for genre in genre_list:

            if genre in row['genres_list']:
                imdb_score = float(row['imdb_score'])
                imdb_votes = float(row['imdb_votes'])
                tmdb_score = float(row['tmdb_score'])
                tmdb_popularity = float(row['tmdb_popularity'])
                runtime = int(row['runtime'])
                movie = 1

                if genre not in data_dict.keys(): # Wanneer data_dict nog niet de genre heeft.
                    popularty_dict = {'imdb_score': [imdb_score], 'imdb_votes': [imdb_votes], 'tmdb_score': [tmdb_score],
                                      'tmdb_popularity': [tmdb_popularity], 'runtime':[runtime], 'movie': movie}
                    data_dict[genre] = popularty_dict

                if genre in data_dict.keys(): # Wanneer data_dict de genre al heeft.
                    popularty_dict = {'imdb_score': imdb_score, 'imdb_votes': imdb_votes, 'tmdb_score': tmdb_score,
                                      'tmdb_popularity': tmdb_popularity, 'runtime':runtime, 'movie': movie}
                    for x in popularty_dict.keys():
                        value = popularty_dict[x]

                        if x == 'movie':
                            data_dict[genre][x] += value
                        else:
                            data_dict[genre][x].append(value)


                else:
                    print('Something is off')

    return pd.DataFrame(data_dict)

def genres_to_list(row):
    genres = row['genres']
    genres_list = genres[1:-1].split(',') # Verkrijgen van aparte genres
    genres_list = [i.replace(' ', '').replace('\'', '') for i in genres_list] # Verwijderen van spaties en quotations.
    row['genres_list'] = genres_list
    return row

netflix = netflix.apply(genres_to_list, axis=1)


genres_list = ['drama', 'thriller', 'european', 'romance', 'scifi', 'action', 'crime', 'music', 'comedy',
               'fantasy', 'family', 'animation', 'documentation', 'horror', 'reality', 'history', 'sport', 'war', 'western']
genre_data = data_per_genre_dict(dataset=netflix, genre_list=genres_list)

st.write('Hieronder zijn het aantal Netflix films per genre afgebeeld. Films met de genre: drama, comedy, thriller, action en romance zijn het meest aanwezig op Netflix. Reality, western, oorlog, sport en geschiedenis films komen minder voor op Netflix.')

data = genre_data.T
x = data.index
y = data['movie'].sort_values(ascending=False)

fig1 = px.bar(data_frame=data, x=x, y=y,
              width=700, height=500,
              title='Aantal films per genre op Netflix', template='simple_white', labels={'y':'Aantal films', 'index': 'Genre'})


fig1.update_traces(hovertemplate = '%{label} <br>Aantal films: %{y}<extra></extra>')
fig1.update_layout(hovermode="closest")

bar_moviecount = fig1
fig1.show()

st.plotly_chart(bar_moviecount, use_container_widtdh=True)



def BoxStipplot(dataset, var, xlabel_text, title_text, log=False):
    sns.set_theme(style="ticks")
    data = dataset.loc[var]

    fig, ax = plt.subplots(figsize=(8, 10), dpi=110)

    if log == True:
        sns.boxplot(data=data, orient='h', showfliers=False,
                    width=.6, palette="vlag").set(xscale='log')

        sns.stripplot(data=data, orient='h', alpha=0.4,
                      size=4, color=".3", linewidth=0, ).set(xscale='log')

    else:
        sns.boxplot(data=data, orient='h', showfliers=False,
                    width=.6, palette="vlag")

        sns.stripplot(data=data, orient='h', alpha= 0.4,
                      size=4, color=".3", linewidth=0)


    ax.set_yticklabels(['drama', 'thriller', 'action', 'crime', 'romance', 'comedy', 'fantasy',
                        'horror', 'history', 'sport', 'documentation', 'scifi', 'family',
                        'animation', 'western', 'european', 'music', 'war'], fontdict={'fontsize': 11})

    sns.despine(trim=True, left=True)
    ax.xaxis.grid(True)
    ax.set_xlabel(xlabel_text)
    ax.set_title(title_text)

    return fig

boxstipplot_runtime = BoxStipplot(genre_data, 'runtime', 'Tijdsduur in min', 'Tijdsduur film per genre')
boxstipplot_imdb_score = BoxStipplot(genre_data, 'imdb_score', 'IMDb score', 'IMDb score per genre')
boxstipplot_tmdb_score = BoxStipplot(genre_data, 'tmdb_score', 'TMDb score', 'TMDb score per genre')
boxstipplot_tmdb_popularity = BoxStipplot(genre_data, 'tmdb_popularity', 'TMDb popularity (log)', 'TMDb popularity per genre', log=True)


st.markdown("""
In onderstaand figuur zijn boxplots weergegeven van de gemiddelde tijdsduur, populariteit, en scores van Netflix films per genre. Zoals het <histogram> al weergaf, duren de meeste Netflix films zo’n 90 tot 120 min. Met dit figuur kunnen we nu inzien dat veel Netflix films met een tijdduur van 30 tot 40 min, afkomstig zijn van de genres: comedy, fantasy, family, animation, en documentation. De significante spike aan Netflix films in het <histogram> rond 60 min kunnen we in de boxplot koppelen aan de genres: comedy, documentation, en animation.

In de boxplot is te zien dat de gemiddelde documentaire niet langer duurt dan circa 90 min. Het is mogelijk lastiger om de aandacht van een kijker te behouden bij een langere documentaire. Wanneer we naar de TMDb scores en populariteit kijken, zien we dat documentaires een relatief hoge score krijgen ten opzichte van andere genres, en is de populariteit van documentaires gemiddeld het laagst. Het blijkt dus dat er veel goede documentaires zijn, alleen worden deze niet bekeken.

Een ander genre dat uitspringt is horror. Een gemiddelde horrorfilm duurt niet langer dan 100 min en heeft gemiddeld de laagste TMDb score ten opzichte van andere genres. Ook is horror één van het meest populaire genres die gekeken wordt. Het is begrijpelijk dat een horror film niet de tijdsduur heeft dat vergelijkbaar is met een drama- of actiefilm, dat zou immer veel vergen van de kijker. Een horror verhoogd namelijk het stress niveau door middel van angst en kan zorgen voor vermoeidheid, slapeloosheid, en concentratieproblemen [[https://www.sciencedirect.com/science/article/pii/S1053811920300094?via%3Dihub,](https://www.sciencedirect.com/science/article/pii/S1053811920300094?via%3Dihub) [https://www.unitedconsumers.com/blog/gezondheid/gevolgen-van-stress.jsp](https://www.unitedconsumers.com/blog/gezondheid/gevolgen-van-stress.jsp)]. Het blijkt dus dat veel mensen graag horrorfilms kijken, alleen zijn er maar weinig die daadwerkelijk een goede horrorfilm zijn.
""")

st.pyplot(boxstipplot_runtime, use_container_widtdh=True)
st.pyplot(boxstipplot_tmdb_score, use_container_widtdh=True)
st.pyplot(boxstipplot_tmdb_popularity, use_container_widtdh=True)



def set_genre(row):
    row['genre1'] = row['genres_list'][0]
    return row

netflix = netflix.apply(set_genre, axis=1)

st.markdown("""
Hieronder is een scatterplot weergegeven van de TMDb score en populariteit van netflix films in de netflixdataset. Met de multiselect tool kun je een scatter plot maken van de verschillende genres.
""")

data = netflix

# Misschien moet de kolom genres specifiek worden aangegeven.
option = st.multiselect("Genres", data)
plot_data = netflix[netflix['genre1'].isin(option)]

fig2 = px.scatter(data_frame=plot_data, x='tmdb_popularity', y='tmdb_score', log_x=True, color='genre1',
                  width=900, height=700, opacity=0.5, title='De TMDb score per film tegen de populariteit', labels={'tmdb_score':'TMDb Score','tmdb_popularity' :'TMDb Populariteit'}, template='none', custom_data=['title', 'genre1'])

fig2.update_traces(hovertemplate="<br>".join(["Movie title: %{customdata[0]}", "TMDb populariteit: %{x}", "TMDb populariteit: %{y}", '<extra></extra>']))

fig2.update_layout(hovermode="closest")

scatter_score_pop = fig2
fig2.show()

st.plotly_chart(scatter_score_pop, use_container_widtdh=True)
