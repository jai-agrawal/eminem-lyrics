# imports
import lyricsgenius as lg
import pickle
import json
import os

# Data Processing

# We will first be pulling data for all of Eminem's songs and gathering the albums, songs thereof
# For the same, the lyricsgenius library will be used, which pulls data from the Genius website

API_KEY = ''
# Calling for album-wise data from Genius
genius = lg.Genius(API_KEY, skip_non_songs=True,
                   excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
artist_name = 'Eminem'
album_list = ['Infinite', 'The Slim Shady LP', 'The Marshall Mathers LP', 'The Eminem Show', 'Encore', 'Relapse',
              'Recovery', 'The Marshall Mathers LP 2', 'Revival', 'Kamikaze', 'Music to Be Murdered By']

# Saving lyrical data to JSON files
for album in album_list:
    try:
        album = genius.search_album(album, artist_name)
        album.save_lyrics()
    except:
        pass

# Using JSON files to create list:
# Instantiating list of dictionaries to hold data
albums = []

# Filling dict with lyrical data
for filename in os.listdir():
    if filename.endswith('.json'):
        album = json.load(open(f'{filename}'))
        album_name = album['name']
        release_year = album['release_date_components']['year']
        song_titles = []
        song_lyrics = []
        for i in range(0, len(album['tracks']), 1):
            song_titles.append(album['tracks'][i]['song']['title'])
            song_lyrics.append(album['tracks'][i]['song']['lyrics'])
        albums.append({'Name' : album_name, 'ReleaseYear': release_year, 'Songs': {'Title': song_titles,
                                                                                   'Lyrics': song_lyrics}})

# For the purposes of topic modelling, we need only the lyrical data. Further album-wise analysis will require the
# dictionary as a whole.
lyrics = []

for album in albums:
    for songs_lyrics in album['Songs']['Lyrics']:
        lyrics.append(songs_lyrics)

# Remove nan values
lyrics = [x for x in lyrics if str(x) != 'nan']

with open('lyrics.pkl', 'wb') as f:
    pickle.dump(lyrics, f)