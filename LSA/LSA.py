# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 22:40:57 2016

@author: rsk
"""

import nltk
from nltk.corpus import stopwords
import re
from string import *
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
import pandas as pd

import numpy as np

#%%

stemmer = PorterStemmer()

dataFile = open("/home/rsk/Documents/Text mining/test.txt","rb")
data = dataFile.read()
data = data.split("\n")              #Extracting the individual sentences

"""
For the headlines.txt dataset, since all sentences start and end with square brackets, we use 
   texts.append(data[i][2:-2])
Otherwise, we use
   texts.append(data[i])

"""

texts=[]
for i in range(len(data)):
    texts.append(data[i])      # Removing the square brackets present in the text file
    texts[i] = re.sub("-"," ",texts[i])         # substituting "-" with " "
    texts[i] = texts[i].translate(None,string.punctuation).lower()
    texts[i] = nltk.word_tokenize(texts[i])
    texts[i] = [stemmer.stem(word) for word in texts[i] if not word in stopwords.words('english')]
    texts[i] = join(texts[i]," ")


#%% 
    
"""
NOTE : TfidfVectorizer is a more advanced version of TfidfTransformer. TfidfTransformer takes as input
a term document matrix and computes the tf-idf weights. However TfidfVectorizer, take a corpus, vectorizes
it and then computes the tf-idf weights. So its more convenient than TfidfTransformer
"""    
    
transformer = TfidfVectorizer()              #Initialize the TF-IDF vectorizer
tfidf = transformer.fit_transform(texts)     #Fit the headline text to the tf-idf model

"""
NOTE : transfomer.fit_transform and transoformer.transform are different. While the former is used to 
initialize the model , the latter is used when a model already exists and you want to convert a new document
into the format describes by the model. More like fit_transoform is for trainingand transform for testing

"""


#%%

#transformer.get_feature_names()           # To get the list of terms in the model
#transformer.vocabulary_                  # Gives the terms with their corresponding indices in the matrix

dat = pd.DataFrame(tfidf.toarray(),index = data,columns = transformer.get_feature_names())   #Transformming the term-document matrix into a pandas DataFrame


#%%

svd = TruncatedSVD(n_components = 2, algorithm="arpack")
"""
For document-document similarity use tfidf as it is

For term-term similarity, use tfidf.T

The TruncatedSVD algorithm returns the components for only the rows of the input matrix. Therefore
we have to take the transpose of the tfidf matrix to get the components of 

"""

lsa = svd.fit_transform(tfidf.T)



#%%

"""
get_feature_names() outputs the terms in the same order as they are in the term-document matrix

"""

def getClosestTerm(term,transformer,model):
    """
    Outputs the closest term to a given term based on the transformer and givem model
    
    transformer : The tf-idf transformer which contains info about the term names and their indices
    
    model : The LSA model that contains the components for each term
    
    """
    
    term = stemmer.stem(term)         # Because the term-document matrix only has stemmed terms
    index = transformer.vocabulary_[term]       # find the index corresponding to the term
    
    
    model = np.dot(model,model.T)       #Computes the dot product for all term-term pairs
    
    # We remove the term corresponding to the term itself as it would have the highest dot product value
    # Since we are removing a term the argmax function can also return the wrong index. Hence we use the 
    # if condition to return the right index
    searchSpace =np.concatenate( (model[index][:index] , model[index][(index+1):]) )  
    
    out = np.argmax(searchSpace)
    
    
    if out<index:    
        return transformer.get_feature_names()[out]    
    else:
        return transformer.get_feature_names()[(out+1)]
        
    
def kClosestTerms(k,term,transformer,model):
    
    term = stemmer.stem(term)
    index = transformer.vocabulary_[term]
    
    model = np.dot(model,model.T)
    
    
    closestTerms = {}
    for i in range(len(model)):
        closestTerms[transformer.get_feature_names()[i]] = model[index][i]
        
    sortedList = sorted(closestTerms , key= lambda l : closestTerms[l])
    
    
    return sortedList[::-1][0:k]


#%%

getClosestTerm("boy",transformer,lsa)
kClosestTerms(5,"boy",transformer,lsa)

#%%



