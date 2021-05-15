# This script is intended to analyse Eminem's lyrics using Natural Language Processing and Machine Learning

# imports
import pickle
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from utils import print_top_words

with open('lyrics.pkl', 'rb') as f:
    lyrics = pickle.load(f)

# Defining stop-words
stop_words = list(stopwords.words('english'))
# extending stop_words, based on observations
stop_words.extend(["i\'m", 'i\'ll', 'ya', 'u', "'em", "i'ma", "yo'", 'like', 'cause', 'um', 'alright', 'ooh',
                   'ha', 'get', 'go', 'shit', 'fuck', 'fuckin','fucking', 'shit', 'got', 'see', 'sayin',
                   'huh', 'oh', 'got', 'yo', 'yah', 'til', 'la', 'em', 'ah', 'going', 'really', 'gonna', 'yeah'
                   ])

# Building Models

# Bag-Of-Words
cvec = CountVectorizer(min_df=5,
                       max_df=0.95,
                       stop_words=stop_words,
                       ngram_range=(1,1),
                      )
bow = cvec.fit_transform(lyrics)

# TF-IDF
tvec = TfidfVectorizer(min_df=5,
                       max_df=0.95,
                       stop_words=stop_words,
                       ngram_range=(1,3),
                      )
tfidf = tvec.fit_transform(lyrics)

# Topic Modelling

n_top_words = 7 # Set number

bow_feature_names = cvec.get_feature_names()
tfidf_feature_names = tvec.get_feature_names()

# NMF with TFIDF
nmf2 = NMF(n_components=5, random_state=42,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.01,
          l1_ratio=.5).fit(tfidf)
print("\nTopics in NMF model (tfidf):")
print_top_words(nmf2, tfidf_feature_names, n_top_words)
