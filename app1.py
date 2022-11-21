import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

movies = pkl.load(open('./data/movies.pkl','rb'))
genome = pkl.load(open('./data/genome.pkl','rb'))
tags = pkl.load(open('./data/tags.pkl','rb'))
weights = pkl.load(open('./data/weights.pkl', 'rb'))

np_genome = genome.to_numpy()
np_movies = movies.to_numpy()


def create_id(np_movies, genome):
    movies_id = {}
    for index, movie in enumerate(np_movies):
        movies_id[movie[1]] = (movie[0], index)
    tags_id = {}
    for index, tag in enumerate(np.unique(genome.tag)):
        tags_id[tag] = (np.unique(genome.tagId)[index],index)
    return movies_id, tags_id
movies_id, tags_id = create_id(np_movies, genome)
def rel_item(item,tag):
    movie_id = movies_id[item][0]
    tag_id = tags_id[tag][0]
    return float(np_genome[(np.where((np_genome[:,0] == movie_id) & (np_genome[:,1] == tag_id)))][:,2])
def rel_all_items(tag):
    tag_index = tags_id[tag][1]
    len_movie = len(movies)
    rel_items = np_genome[tag_index*len_movie:tag_index*len_movie+len_movie,2].tolist()
    return rel_items
def linear_sat(item,tag,direction):
    a = (rel_all_items(tag) - np.array(rel_item(item,tag)))*direction
    result = np.clip(a, 0, 1)
    return result
def diminish_sat(item,tag,direction):
    critique_dist = linear_sat(item,tag,direction)
    result = 1 - np.exp(-5*critique_dist)
    return result
def tag_item_vector(item):
    item_id = movies_id[item][0]
    full_tags_item = [np_genome[i][2] for i in np.where(np_genome[:,0] == item_id)[0]]  
    return full_tags_item
def cos_similarity(item, weights):
    
    # Create all movies's tags scores
    list_tags = np.zeros((len(movies),len(tags_id)))

    for index,tag_score in enumerate(np_genome[:,2]):
        tags_index = index // len(movies)
        movies_index = index - tags_index*len(movies)
        list_tags[movies_index,tags_index] = tag_score
        
    # Extract 1 query movie from matrix above 
    query_index = movies_id[item][1]
    query_tags = list_tags[query_index]
    
    numerator = np.dot(list_tags * list_tags[query_index],weights)
    
    denominator = np.sqrt(np.dot(list_tags**2, weights)) * np.sqrt(np.dot(query_tags**2, weights))
    
    result = numerator / denominator
    return result
def encode_movies():
    genres = {}
    index = 0
    movies_encode = []
    for movie in np_movies[:,2]:
        movie_encode = [0]*20
        for genre in movie.split('|'):
            if genre not in genres:
                genres[genre] = index
                index += 1
            movie_encode[genres[genre]] = 1
        movies_encode.append(movie_encode)
    return movies_encode
movies_encode = encode_movies()
def genres_similarity(item):
    index = movies_id[item][1]
    numerator = np.dot(movies_encode, movies_encode[index])
    denominator = np.sqrt(np.square(movies_encode[index]).sum()) * np.sqrt(np.sum(np.square(movies_encode),axis=1))
    return numerator/denominator
def recommendation(item, list_critiques):
    a = 0
    a = 0
    if item not in movies_id:
        print('Have not the film in system')
        return
    for critique in list_critiques:
        tag = critique[0]
        direction = critique[1]
        value = diminish_sat(item,tag,direction)
        cos = cos_similarity(item, weights)
        genre_simi = genres_similarity(item)
        a += value * cos * genre_simi
    score = {}
    for index, movie in enumerate(np_movies[:,1]):
        score[movie] = a[index]   
    result = sorted(score.items(), key=lambda x:x[1], reverse=True)[:10]
    return result


st.set_page_config(
    page_title="Movie Recommendation System",
    initial_sidebar_state="expanded",
)



ListMovies = movies['title'].to_list()
st.title('Movie Recommendation System')
movie = st.selectbox("Movie selection: ", options = list(ListMovies), label_visibility= 'visible')
# movie = st.text_input('Enter selection: ')
# AnalystButton = st.button('Analyst')
col1, col2, col3, col4, col5 = st.columns(5)
ListDirec = []
ListTag = ['action', 'violence', 'funny', 'fantasy', 'comedy', 'classic', 'romance', 'sci-fi',  'atmospheric', 'based on a book']
Lv = {
    'Less' : -0.5,
    'Ok'   : 1,
    'More' : 0.5
}

with col1: 
    st.subheader('Action')
    ActionBar = st.progress(0)
    if movie :
        ActionBar.progress(rel_item(movie, ListTag[0]))

    ActionLv = st.radio('', ['Less', 'Ok', 'More'], key= 'Action', index= 1, label_visibility= 'visible')
    ListDirec.append(('action', Lv.get(ActionLv)))

    st.subheader('Violence')
    ViolenceBar = st.progress(0)
    if movie :
        ViolenceBar.progress(rel_item(movie, ListTag[1]))

    ViolentLv = st.radio('', ['Less', 'Ok', 'More'], key= 'Violence', index= 1, label_visibility= 'visible')
    ListDirec.append(('violence', Lv.get(ViolentLv)))

with col2:
    st.subheader('Funny')
    FunnyBar = st.progress(0)
    if movie :
        FunnyBar.progress(rel_item(movie, ListTag[2]))

    FunnyLv = st.radio('', ['Less', 'Ok', 'More'],key= 'Funny', index= 1, label_visibility= 'visible')
    ListDirec.append(('funny', Lv.get(FunnyLv)))

    st.subheader('Fantasy')
    FantasyBar = st.progress(0)
    if movie :
        FantasyBar.progress(rel_item(movie, ListTag[3]))

    FantasyLv = st.radio('', ['Less', 'Ok', 'More'],key= 'Fantasy', index= 1, label_visibility= 'visible')
    ListDirec.append(('fantasy', Lv.get(FantasyLv)))

with col3: 
    st.subheader('Comedy')
    ComedyBar = st.progress(0)
    if movie :
        ComedyBar.progress(rel_item(movie, ListTag[4]))

    ComedyLv = st.radio('', ['Less', 'Ok', 'More'],key= 'Comedy', index= 1, label_visibility= 'visible')
    ListDirec.append(('comedy', Lv.get(ComedyLv)))

    st.subheader('Classic')
    ClassicBar = st.progress(0)
    if movie :
        ClassicBar.progress(rel_item(movie, ListTag[5]))

    ClassicLv = st.radio('', ['Less', 'Ok', 'More'],key= 'Classic', index= 1, label_visibility= 'visible')
    ListDirec.append(('classic', Lv.get(ClassicLv)))
with col4: 
    st.subheader('Romance')
    RomanceBar = st.progress(0)
    if movie :
        RomanceBar.progress(rel_item(movie, ListTag[6]))

    RomanceLv = st.radio('', ['Less', 'Ok', 'More'],key= 'Romance', index= 1, label_visibility= 'visible')
    ListDirec.append(('romance', Lv.get(RomanceLv)))

    st.subheader('Sci-fi')
    ScifiBar = st.progress(0)
    if movie :
        ScifiBar.progress(rel_item(movie, ListTag[7]))

    ScifiLv = st.radio('', ['Less', 'Ok', 'More'],key= 'Sci-fi', index= 1, label_visibility= 'visible')
    ListDirec.append(('sci-fi', Lv.get(ScifiLv)))
with col5:
    st.subheader('Atmospheric')
    AtmosphericBar = st.progress(0)
    if movie :
        AtmosphericBar.progress(rel_item(movie, ListTag[8]))

    AtmosphericLv = st.radio('', ['Less', 'Ok', 'More'],key= 'Atmospheric', index= 1, label_visibility= 'visible')
    ListDirec.append(('atmospheric', Lv.get(AtmosphericLv)))

    st.subheader('BaseOnBook')
    BaseOnBookBar = st.progress(0)
    if movie :
        BaseOnBookBar.progress(rel_item(movie, ListTag[9]))

    BaseOnBookLv = st.radio('', ['Less', 'Ok', 'More'],key= 'BaseOnBook', index= 1, label_visibility= 'visible')
    ListDirec.append(('based on a book', Lv.get(BaseOnBookLv)))
import time
if st.button('Recommend movie'):
    result = recommendation(movie, ListDirec)
    for movie in result:
        st.write(movie[0])
