
# coding: utf-8

# In[1]:

# Import Natural Language Toolkit Library and various packages to manipulate text

import nltk # Primary ntlk platform
import pandas as pd # Used because Pandas
import string # Used to remove punctuation
from nltk.corpus import stopwords # Used to remove stopwords
from nltk.stem.porter import PorterStemmer # Used for stemming
from nltk import word_tokenize # Used to tokenize words to make manipulation easier
from nltk.util import ngrams # Used for producing ngrams and their associated frequencies


# In[2]:

# Save filenames for ingestion in the next step

filenames = ['6334220.txt', '6334221.txt', '6334222.txt', '6334223.txt', '6334224.txt', '6334225.txt', '6334226.txt',
              '6334227.txt', '6334228.txt', '6334229.txt']


# In[3]:

stopset = stopwords.words('english') # Store stopwords in a variable
exclude = set(string.punctuation) # Store punctuation in a variable
word_list = [] # Create an empty list to store words from text files. 

#Note that this will produce a list of 10 lists, each representing one doc.

for item in filenames: # Loop through filenames
    with open(item, 'r') as text_file: # Open each file as a text file
        text = text_file.read() # Read in the next
        tokens=word_tokenize(str(text)) # Tokenize the words as strings
        tokens = [w for w in tokens if not w in stopset] # Remove stopwords
        tokens = [w for w in tokens if not w in exclude] # Remove punctuation
        word_list.append(tokens) # Append remaining tokenized words to list


# In[4]:

st = PorterStemmer() # Save stemming function as an object. Note that our stemming is based on the Porter Stemming Module.

# List comprehension that loops through each word and stems it according to the Porter algorithm
stemmed_words = [[st.stem(word) for word in sentence] for sentence in word_list]

# List comprehension that then loops through each word and encodes it in utf-8. This is useful for getting rid of the annoying
# 'u' that appears before each string otherwise
stemmed_words = [[word.encode("utf-8") for word in sentence] for sentence in word_list]


# In[5]:

# List comprehension that flattens the 10 sublists into one aggregate list, useful for computing summary ngrams later on

flat_stemmed_words = [item for sublist in stemmed_words for item in sublist]


# In[6]:

# Produce uni, bi, and trigrams from the combined 10 documents

full_doc_unigrams = list(ngrams(flat_stemmed_words, 1))
full_doc_bigrams = list(ngrams(flat_stemmed_words, 2))
full_doc_trigrams = list(ngrams(flat_stemmed_words, 3))


# In[7]:

# Calculate frequencies of summary ngrams produced in previous step

full_doc_unidist = nltk.FreqDist(full_doc_unigrams)
full_doc_bidist = nltk.FreqDist(full_doc_bigrams)
full_doc_tridist = nltk.FreqDist(full_doc_trigrams)


# In[8]:

unigrams = [] # Empty list to store unigrams
bigrams = [] # Empty list to store bigrams
trigrams = [] # Empty list to store trigrams

# Loop that will produce uni, bi, and trigrams for all words in each of the 10 documents

# End result will be 3 lists, each containing 10 sublists representing uni, bi, and trigrams for each document

for word in stemmed_words:
    uni = ngrams(word, 1) # Create ngrams
    bi = ngrams(word, 2) 
    tri = ngrams(word, 3) 
    
    uni_gram = list(uni) # Store ngrams in a list
    bi_gram = list(bi)
    tri_gram = list(tri)
    
    unigrams.append(uni_gram) # Append ngram sublists to master list
    bigrams.append(bi_gram) 
    trigrams.append(tri_gram) 


# In[9]:

unidist = [] # Empty list to store unigram frequencies
bidist = [] # Empty list to store bigram frequencies
tridist = [] # Empty list to store trigram frequencies

# Three loops that calculate frequencies of all uni, bi, and trigrams in the 10 documents and store them in a list

for gram in unigrams:
    dist = nltk.FreqDist(gram)
    unidist.append(dist)
    
for gram in bigrams:
    dist = nltk.FreqDist(gram)
    bidist.append(dist)
    
for gram in trigrams:
    dist = nltk.FreqDist(gram)
    tridist.append(dist)


# In[10]:

# The remaining code is entirely devoted to formatting the data before writing to CSV

# Remove commas, parentheses, and apostrophes from all stored unigrams, bigrams, and trigrams

df_summ_uni = pd.DataFrame(full_doc_unidist.items(), columns = (['Unigram', 'Count']))
df_summ_uni['Unigram'] = df_summ_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df_summ_uni['Unigram'] = df_summ_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df_summ_uni['Unigram'] = df_summ_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df_summ_uni['Unigram'] = df_summ_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df_summ_bi = pd.DataFrame(full_doc_bidist.items(), columns = (['Bigram', 'Count']))
df_summ_bi['Bigram'] = df_summ_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df_summ_bi['Bigram'] = df_summ_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df_summ_bi['Bigram'] = df_summ_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df_summ_bi['Bigram'] = df_summ_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df_summ_tri = pd.DataFrame(full_doc_tridist.items(), columns = (['Trigram', 'Count']))
df_summ_tri['Trigram'] = df_summ_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df_summ_tri['Trigram'] = df_summ_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df_summ_tri['Trigram'] = df_summ_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df_summ_tri['Trigram'] = df_summ_tri['Trigram'].map(lambda x: str(x).replace("'", ''))


# In[11]:

df1_uni = pd.DataFrame(unidist[0].items(), columns = (['Unigram', 'Count']))
df1_uni['Unigram'] = df1_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df1_uni['Unigram'] = df1_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df1_uni['Unigram'] = df1_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df1_uni['Unigram'] = df1_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df2_uni = pd.DataFrame(unidist[1].items(), columns = (['Unigram', 'Count']))
df2_uni['Unigram'] = df2_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df2_uni['Unigram'] = df2_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df2_uni['Unigram'] = df2_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df2_uni['Unigram'] = df2_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df3_uni = pd.DataFrame(unidist[2].items(), columns = (['Unigram', 'Count']))
df3_uni['Unigram'] = df3_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df3_uni['Unigram'] = df3_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df3_uni['Unigram'] = df3_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df3_uni['Unigram'] = df3_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df4_uni = pd.DataFrame(unidist[3].items(), columns = (['Unigram', 'Count']))
df4_uni['Unigram'] = df4_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df4_uni['Unigram'] = df4_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df4_uni['Unigram'] = df4_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df4_uni['Unigram'] = df4_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df5_uni = pd.DataFrame(unidist[4].items(), columns = (['Unigram', 'Count']))
df5_uni['Unigram'] = df5_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df5_uni['Unigram'] = df5_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df5_uni['Unigram'] = df5_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df5_uni['Unigram'] = df5_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df6_uni = pd.DataFrame(unidist[5].items(), columns = (['Unigram', 'Count']))
df6_uni['Unigram'] = df6_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df6_uni['Unigram'] = df6_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df6_uni['Unigram'] = df6_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df6_uni['Unigram'] = df6_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df7_uni = pd.DataFrame(unidist[6].items(), columns = (['Unigram', 'Count']))
df7_uni['Unigram'] = df7_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df7_uni['Unigram'] = df7_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df7_uni['Unigram'] = df7_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df7_uni['Unigram'] = df7_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df8_uni = pd.DataFrame(unidist[7].items(), columns = (['Unigram', 'Count']))
df8_uni['Unigram'] = df8_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df8_uni['Unigram'] = df8_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df8_uni['Unigram'] = df8_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df8_uni['Unigram'] = df8_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df9_uni = pd.DataFrame(unidist[8].items(), columns = (['Unigram', 'Count']))
df9_uni['Unigram'] = df9_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df9_uni['Unigram'] = df9_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df9_uni['Unigram'] = df9_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df9_uni['Unigram'] = df9_uni['Unigram'].map(lambda x: str(x).replace("'", ''))

df10_uni = pd.DataFrame(unidist[9].items(), columns = (['Unigram', 'Count']))
df10_uni['Unigram'] = df10_uni['Unigram'].map(lambda x: str(x).replace(',', ''))
df10_uni['Unigram'] = df10_uni['Unigram'].map(lambda x: str(x).replace(')', ''))
df10_uni['Unigram'] = df10_uni['Unigram'].map(lambda x: str(x).replace('(', ''))
df10_uni['Unigram'] = df10_uni['Unigram'].map(lambda x: str(x).replace("'", ''))


# In[12]:

df1_bi = pd.DataFrame(bidist[0].items(), columns = (['Bigram', 'Count']))
df1_bi['Bigram'] = df1_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df1_bi['Bigram'] = df1_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df1_bi['Bigram'] = df1_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df1_bi['Bigram'] = df1_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df2_bi = pd.DataFrame(bidist[1].items(), columns = (['Bigram', 'Count']))
df2_bi['Bigram'] = df2_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df2_bi['Bigram'] = df2_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df2_bi['Bigram'] = df2_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df2_bi['Bigram'] = df2_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df3_bi = pd.DataFrame(bidist[2].items(), columns = (['Bigram', 'Count']))
df3_bi['Bigram'] = df3_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df3_bi['Bigram'] = df3_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df3_bi['Bigram'] = df3_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df3_bi['Bigram'] = df3_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df4_bi = pd.DataFrame(bidist[3].items(), columns = (['Bigram', 'Count']))
df4_bi['Bigram'] = df4_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df4_bi['Bigram'] = df4_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df4_bi['Bigram'] = df4_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df4_bi['Bigram'] = df4_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df5_bi = pd.DataFrame(bidist[4].items(), columns = (['Bigram', 'Count']))
df5_bi['Bigram'] = df5_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df5_bi['Bigram'] = df5_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df5_bi['Bigram'] = df5_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df5_bi['Bigram'] = df5_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df6_bi = pd.DataFrame(bidist[5].items(), columns = (['Bigram', 'Count']))
df6_bi['Bigram'] = df6_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df6_bi['Bigram'] = df6_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df6_bi['Bigram'] = df6_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df6_bi['Bigram'] = df6_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df7_bi = pd.DataFrame(bidist[6].items(), columns = (['Bigram', 'Count']))
df7_bi['Bigram'] = df7_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df7_bi['Bigram'] = df7_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df7_bi['Bigram'] = df7_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df7_bi['Bigram'] = df7_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df8_bi = pd.DataFrame(bidist[7].items(), columns = (['Bigram', 'Count']))
df8_bi['Bigram'] = df8_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df8_bi['Bigram'] = df8_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df8_bi['Bigram'] = df8_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df8_bi['Bigram'] = df8_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df9_bi = pd.DataFrame(bidist[8].items(), columns = (['Bigram', 'Count']))
df9_bi['Bigram'] = df9_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df9_bi['Bigram'] = df9_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df9_bi['Bigram'] = df9_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df9_bi['Bigram'] = df9_bi['Bigram'].map(lambda x: str(x).replace("'", ''))

df10_bi = pd.DataFrame(bidist[9].items(), columns = (['Bigram', 'Count']))
df10_bi['Bigram'] = df10_bi['Bigram'].map(lambda x: str(x).replace(',', ''))
df10_bi['Bigram'] = df10_bi['Bigram'].map(lambda x: str(x).replace(')', ''))
df10_bi['Bigram'] = df10_bi['Bigram'].map(lambda x: str(x).replace('(', ''))
df10_bi['Bigram'] = df10_bi['Bigram'].map(lambda x: str(x).replace("'", ''))


# In[13]:

df1_tri = pd.DataFrame(tridist[0].items(), columns = (['Trigram', 'Count']))
df1_tri['Trigram'] = df1_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df1_tri['Trigram'] = df1_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df1_tri['Trigram'] = df1_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df1_tri['Trigram'] = df1_tri['Trigram'].map(lambda x: str(x).replace("'", ''))

df2_tri = pd.DataFrame(tridist[1].items(), columns = (['Trigram', 'Count']))
df2_tri['Trigram'] = df2_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df2_tri['Trigram'] = df2_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df2_tri['Trigram'] = df2_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df2_tri['Trigram'] = df2_tri['Trigram'].map(lambda x: str(x).replace("'", ''))

df3_tri = pd.DataFrame(tridist[2].items(), columns = (['Trigram', 'Count']))
df3_tri['Trigram'] = df3_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df3_tri['Trigram'] = df3_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df3_tri['Trigram'] = df3_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df3_tri['Trigram'] = df3_tri['Trigram'].map(lambda x: str(x).replace("'", ''))

df4_tri = pd.DataFrame(tridist[3].items(), columns = (['Trigram', 'Count']))
df4_tri['Trigram'] = df4_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df4_tri['Trigram'] = df4_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df4_tri['Trigram'] = df4_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df4_tri['Trigram'] = df4_tri['Trigram'].map(lambda x: str(x).replace("'", ''))

df5_tri = pd.DataFrame(tridist[4].items(), columns = (['Trigram', 'Count']))
df5_tri['Trigram'] = df5_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df5_tri['Trigram'] = df5_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df5_tri['Trigram'] = df5_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df5_tri['Trigram'] = df5_tri['Trigram'].map(lambda x: str(x).replace("'", ''))

df6_tri = pd.DataFrame(tridist[5].items(), columns = (['Trigram', 'Count']))
df6_tri['Trigram'] = df6_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df6_tri['Trigram'] = df6_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df6_tri['Trigram'] = df6_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df6_tri['Trigram'] = df6_tri['Trigram'].map(lambda x: str(x).replace("'", ''))

df7_tri = pd.DataFrame(tridist[6].items(), columns = (['Trigram', 'Count']))
df7_tri['Trigram'] = df7_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df7_tri['Trigram'] = df7_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df7_tri['Trigram'] = df7_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df7_tri['Trigram'] = df7_tri['Trigram'].map(lambda x: str(x).replace("'", ''))

df8_tri = pd.DataFrame(tridist[7].items(), columns = (['Trigram', 'Count']))
df8_tri['Trigram'] = df8_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df8_tri['Trigram'] = df8_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df8_tri['Trigram'] = df8_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df8_tri['Trigram'] = df8_tri['Trigram'].map(lambda x: str(x).replace("'", ''))

df9_tri = pd.DataFrame(tridist[8].items(), columns = (['Trigram', 'Count']))
df9_tri['Trigram'] = df9_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df9_tri['Trigram'] = df9_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df9_tri['Trigram'] = df9_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df9_tri['Trigram'] = df9_tri['Trigram'].map(lambda x: str(x).replace("'", ''))

df10_tri = pd.DataFrame(tridist[9].items(), columns = (['Trigram', 'Count']))
df10_tri['Trigram'] = df10_tri['Trigram'].map(lambda x: str(x).replace(',', ''))
df10_tri['Trigram'] = df10_tri['Trigram'].map(lambda x: str(x).replace(')', ''))
df10_tri['Trigram'] = df10_tri['Trigram'].map(lambda x: str(x).replace('(', ''))
df10_tri['Trigram'] = df10_tri['Trigram'].map(lambda x: str(x).replace("'", ''))


# In[14]:

# Write uni, bi, and trigrams to CSV

df1_uni.to_csv("6334220_unigrams.csv", index = False)
df2_uni.to_csv("6334221_unigrams.csv", index = False)
df3_uni.to_csv("6334222_unigrams.csv", index = False)
df4_uni.to_csv("6334223_unigrams.csv", index = False)
df5_uni.to_csv("6334224_unigrams.csv", index = False)
df6_uni.to_csv("6334225_unigrams.csv", index = False)
df7_uni.to_csv("6334226_unigrams.csv", index = False)
df8_uni.to_csv("6334227_unigrams.csv", index = False)
df9_uni.to_csv("6334228_unigrams.csv", index = False)
df10_uni.to_csv("6334229_unigrams.csv", index = False)


# In[15]:

df1_bi.to_csv("6334220_bigrams.csv", index = False)
df2_bi.to_csv("6334221_bigrams.csv", index = False)
df3_bi.to_csv("6334222_bigrams.csv", index = False)
df4_bi.to_csv("6334223_bigrams.csv", index = False)
df5_bi.to_csv("6334224_bigrams.csv", index = False)
df6_bi.to_csv("6334225_bigrams.csv", index = False)
df7_bi.to_csv("6334226_bigrams.csv", index = False)
df8_bi.to_csv("6334227_bigrams.csv", index = False)
df9_bi.to_csv("6334228_bigrams.csv", index = False)
df10_bi.to_csv("6334229_bigrams.csv", index = False)


# In[16]:

df1_tri.to_csv("6334220_trigrams.csv", index = False)
df2_tri.to_csv("6334221_trigrams.csv", index = False)
df3_tri.to_csv("6334222_trigrams.csv", index = False)
df4_tri.to_csv("6334223_trigrams.csv", index = False)
df5_tri.to_csv("6334224_trigrams.csv", index = False)
df6_tri.to_csv("6334225_trigrams.csv", index = False)
df7_tri.to_csv("6334226_trigrams.csv", index = False)
df8_tri.to_csv("6334227_trigrams.csv", index = False)
df9_tri.to_csv("6334228_trigrams.csv", index = False)
df10_tri.to_csv("6334229_trigrams.csv", index = False)


# In[17]:

# Write summary uni, bi, and trigrams to CSV

df_summ_uni.to_csv("summary_unigrams.csv", index = False)
df_summ_bi.to_csv("summary_bigrams.csv", index = False)
df_summ_tri.to_csv("summary_trigrams.csv", index = False)

