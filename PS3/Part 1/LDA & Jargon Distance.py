
# coding: utf-8

# In[5]:

# Import Natural Language Toolkit Library and various packages to manipulate text

import nltk # Primary ntlk platform
import pandas as pd # Used because Pandas
import string # Used to remove punctuation
from nltk.corpus import stopwords # Used to remove stopwords
from nltk.stem.porter import PorterStemmer # Used for stemming
from nltk import word_tokenize # Used to tokenize words to make manipulation easier
from nltk.util import ngrams # Used for producing ngrams and their associated frequencies
from gensim import corpora # Used to generate a corpus from a set of documents
from gensim.models import ldamodel, tfidfmodel # Used for applying Latent Dirichlet Allocation on a corpus of text


# In[2]:

filenames = ['6334220.txt', '6334221.txt', '6334222.txt', '6334223.txt', '6334224.txt', '6334225.txt', '6334226.txt',
              '6334227.txt', '6334228.txt', '6334229.txt']

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


# In[3]:

st = PorterStemmer() # Save stemming function as an object. Note that our stemming is based on the Porter Stemming Module.

# List comprehension that loops through each word and stems it according to the Porter algorithm
stemmed_words = [[st.stem(word) for word in sentence] for sentence in word_list]


# In[6]:

dictionary = corpora.Dictionary(stemmed_words) # Construct word - id mappings from stemmed words

corpus = [dictionary.doc2bow(word) for word in stemmed_words] # Converts document into a bag-of-words format

tfidf = tfidfmodel.TfidfModel(corpus) # Apply tf-idf reduction to produce a list of term frequencies

tfidf_corpus = tfidf[corpus] # Extract terms and their frequencies from tf-idf reduction

lda = ldamodel.LdaModel(tfidf_corpus, id2word=dictionary, num_topics=2, passes = 20) # Apply LDA model, generating 2 topics

topics = [i for i in lda.show_topics()] # Extract topics

print "Topic 1:", topics[0]
print
print "Topic 2:", topics[1]


# In[7]:

filenames = ['lda1.txt', 'lda2.txt', 'lda3.txt', 'lda4.txt', 'lda5.txt']

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


# In[8]:

st = PorterStemmer() # Save stemming function as an object. Note that our stemming is based on the Porter Stemming Module.

# List comprehension that loops through each word and stems it according to the Porter algorithm
stemmed_words = [[st.stem(word) for word in sentence] for sentence in word_list]


# In[9]:

dictionary = corpora.Dictionary(stemmed_words) # Construct word - id mappings from stemmed words

corpus = [dictionary.doc2bow(word) for word in stemmed_words] # Converts document into a bag-of-words format

tfidf = tfidfmodel.TfidfModel(corpus) # Apply tf-idf reduction to produce a list of term frequencies

tfidf_corpus = tfidf[corpus] # Extract terms and their frequencies from tf-idf reduction

lda = ldamodel.LdaModel(tfidf_corpus, id2word=dictionary, num_topics=2, passes = 20) # Apply LDA model, generating 2 topics

topics = [i for i in lda.show_topics()] # Extract topics

print "Topic 1:", topics[0]
print
print "Topic 2:", topics[1]


# In[10]:

books = [stemmed_words[i] for i in (0,1,4)] # Extract documents 1, 2, and 5 into 'Topic A', here considered 'books'
books = [item for sublist in books for item in sublist] # Flatten sublists into one list

languages = [stemmed_words[i] for i in (2,3)] # Extract documents 3 and 4 into 'Topic B', here considered 'languages'
languages = [item for sublist in languages for item in sublist] # Flatten sublists into one list

all_words = [item for sublist in stemmed_words for item in sublist] # Flatten all sublists into one list object


# In[11]:

book_freq = nltk.FreqDist(books) # Calculate the frequency of terms in the book topic set

language_freq = nltk.FreqDist(languages) # Calculate the frequency of terms in the language topic set

all_freq = nltk.FreqDist(all_words) # Calculate the frequency of all words across the five documents


# In[12]:

a = 0.01 # Set teleportation term alpha
Psi = {} # Create an empty dictionary object to store output

for k, v in all_freq.items(): # Select words and their frequency values across all documents
    for i, j in book_freq.items(): # Select words and their frequency values within the book topic subset
        if k in book_freq.keys(): # If the word from the corpus appears in the book topic subset
            # Calculate the cross entropy of that word by multiplying probability of the word in the book topic by 1 - alpha term,
            # then add probability of the word in the entire corpus multiplied by alpha term
            Psi[k] = (1 - a) * (float(j)/len(books)) + a * (float(v)/len(all_words))
        else: # If the word from the corpus does not appear in the book topic subset...
            # Multiply probability of word by alpha term
            Psi[k] = a * (float(v)/len(all_words))


# In[13]:

a = 0.01 # Set teleportation term alpha
Psj = {} # Create an empty dictionary object to store output

for k, v in all_freq.items(): # Select words and their frequency values across all documents
    for i, j in language_freq.items(): # Select words and their frequency values within the language topic subset
        if k in language_freq.keys(): # If the word from the corpus appears in the language topic subset
            # Calculate the cross entropy of that word by multiplying probability of the word in the language topic by 1 - alpha term,
            # then add probability of the word in the entire corpus multiplied by alpha term
            Psj[k] = (1 - a) * (float(j)/len(languages)) + a * (float(v)/len(all_words))
        else: # If the word from the corpus does not appear in the language topic subset...
            # Multiply probability of word by alpha term
            Psj[k] = a * (float(v)/len(all_words))


# In[14]:

import math

# Function for computing Shannon Entropy

def H(prob):
    entropy = 0 # Initialize entropy at zero
    for p_x in prob: # For each word probability...
        # Calculate entropy by multiplying the negative probability of that word by the log (base 2) of its probability
        entropy += - p_x * math.log(p_x, 2)
    return entropy # Return summed entropy


# In[15]:

# Function for computing cross entropy

def Q(prob_i, prob_j):
    entropy = 0 # Initialize entropy at zero
    for key_i, p_ix in prob_i: # For word, word probability in writer topic set
        for key_j, p_jx in prob_j: # For word, word probability in reader topic set
            if key_i == key_j: # Where words are the same...
                # Calculate cross entropy by multiplying the negative probability of the word in topic a,
                # with the log (base 2) probability of the word in topic b
                entropy += - p_ix * math.log(p_jx, 2)
    return entropy # Return summed cross entropy


# In[16]:

# Shannon Entropy of writer set (books)

H = H(Psi.values())

print H


# In[17]:

# Cross Entropy of writer set (books) with reader set (languages)

Q = Q(Psi.items(), Psj.items())

print Q


# In[18]:

# Efficiency of Communication between two topics

E = H / Q

print E


# In[19]:

# Cultural Hole / Jargon Distance between two topics

C = 1 - E

print C

