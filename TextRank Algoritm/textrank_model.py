# load libraries
import numpy as np
import pandas as pd
import nltk
#_>nltk.download('punkt') # one time execution
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


# load data
df = pd.read_csv('G:/rauf/STEPBYSTEP/Projects/NLP/Text Summarization/TextRank Algoritm/my_tennis_article.csv')
#_>print(df['article_text'][0])


# split text into sentences
sentences = []
for s in df['article_text']:
  sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list
#_>print(sentences[:2]) # lets look at two tokenized sentences


# Extract word vectors
word_embeddings = {}
f = open('G:/rauf/STEPBYSTEP/Projects/NLP/Text Summarization/TextRank Algoritm/glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
#_>print(len(word_embeddings)) 


# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


# lets clean stop words
stop_words = stopwords.words('english')
# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


# create vecotrs for sentences
sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)


# create similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


# apply page rank algoritm
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)


# Extract top 10 sentences as the summary
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
for i in range(10):
  print(ranked_sentences[i][1])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
for text summarization task we use text rank algoritm
and we will use the cosine similarity approach for this challenge
'''
