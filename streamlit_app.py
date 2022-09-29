from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np


"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""



with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
    
    
# Opmerking over de code
st.write('Met behulp van de onderstaande code worden de kolommen "budget" en "revenue" uit de movies dataset samengevoegd met de netflix films dataset. Allereerst wordt er een aparte dataset gemaakt met "budget", "revenue" en "id". Vervolgens wordt er met behulp van de kolom "id" die zich in beide datasets bevindt, beide datasets samengevoegd. Vervolgens wordt er nog een nieuwe kolom "winst" toegevoegd door het budget van de omzet af te trekken. Tot slot verwijderen we alle NaN waardes die de waarde 0 hadden gekregen, omdat deze ons niks kunnen vertellen over de data.')

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

st.code(merge_code)
