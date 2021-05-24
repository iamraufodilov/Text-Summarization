# load libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


#get input text
text = """When most people hear the term artificial intelligence, the first thing they usually think of is robots. 
That's because big-budget films and novels weave stories about human-like machines that wreak havoc on Earth. 
But nothing could be further from the truth.
Artificial intelligence is based on the principle that human intelligence can be defined in a way that a machine can easily mimic it and execute tasks, 
from the most simple to those that are even more complex. The goals of artificial intelligence include mimicking human cognitive activity. 
Researchers and developers in the field are making surprisingly rapid strides in mimicking activities such as learning, reasoning, and perception, 
to the extent that these can be concretely defined. 
Some believe that innovators may soon be able to develop systems that exceed the capacity of humans to learn or reason out any subject. 
But others remain skeptical because all cognitive activity is laced with value judgments that are subject to human experience.
"""


# tokenize text
my_stopwords = set(stopwords.words("english"))
words = word_tokenize(text)
#_>print(words)


#create frequency table
freq_table = {}
for w in words:
    w = w.lower()
    if w in my_stopwords:
        continue
    if w in freq_table:
        freq_table[w] += 1
    else:
        freq_table[w] = 1


#create dictionary for sentence
sentences = sent_tokenize(text)
sentence_value = {}
#_>print(sentences)


#
for s in sentences:
    for w, f in freq_table.items():
        if w in s.lower():
            if s in sentence_value:
                sentence_value[s] += 1
            else:
                sentence_value[s] = 1



sum_value = 0
for s in sentence_value:
    sum_value += sentence_value[s]


# average value
average = int(sum_value / len(sentence_value))


# sorting sentence to our summary
summary=''
for s in sentences:
    if (s in sentence_value) and (sentence_value[s]>(1.2 * average)):
        summary += " " + s


print(summary) # very very nice model and summarized pretty good


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION

'''
this model uses extrctive wy to summarize text with the help of nltk library and bunch of functions
this model simple and easy to build.
'''