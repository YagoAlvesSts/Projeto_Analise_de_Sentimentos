# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:54:31 2019

@author: yagoa
"""
import pprint
import pickle
from enelvo import normaliser
import nltk
import string
import spacy
import os
from spacy import tokens
from pathlib import Path


spc = spacy.load('pt_core_news_sm')

with(open("USO_GERAL.p", "rb")) as f:
    all_reviews = pickle.load(f)
#print(all_reviews)



reviews = str(all_reviews)
tratados = []

for review in reviews:
    
    #atribuindo o texto ao modelo spacy
    doc = spc(review)


    #dando split no texto
    doc.text.split()

    tagger = []
    words = []
    for i, word in enumerate(doc):
        if not word.is_punct:
            if not word.is_space:
                #print(word.text, word.pos_)
                tagger = [(word.text, word.pos_)]
                words.append(tagger)
        if i > len(doc):
            break

    print(words)
    tratados.append(words)

arquivo = open('tagger_spacy.txt','w')
arquivo.write(' '.join(map(str, words)))
arquivo.close()

with open(os.path.join("tagger_spacy.p"), "wb") as f:
        pickle.dump(list(words), f)


"""

docu = [token for token in doc]
print("DOCU:\n", docu)
docum = [token.orth_ for token in docu]
print("DOCUM:\n",docum)

#docume = [token.orth_ for token in docum if not token.is_punct]
docume = []
for token in docum :
    if not token.is_punct:
        docume.append(token.orth_)

print("DOCUME:\n",docume)

words = [(token.orth_, token.pos_) for token in docume]

print("\nPOS_TAG:\n", words)



docu = []
#atribuindo os tokens a uma lista
for token in doc:
    docu.append(token.text)




docum = []
for token in doc :
    if not token.is_punct:
        docum.append(token.orth_)

print("\nPOS_TAG:\n", docum)

 
#tira espaço e pontuação
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in docu]

#remove qualquer outro caracter que não seja do alfabeto
words = [word for word in stripped if word.isalpha()]
#print(words)
print("\nPOS_TAG:\n", words)

wrds = [(token.orth_, token.pos_) for token in words]


arquivo = open('tagger_spacy.txt','w')
arquivo.write(' '.join(map(str, doc)))
arquivo.close()

with open(os.path.join("tagger_spacy.p"), "wb") as f:
        pickle.dump(list(doc), f)
        
with(open("tagger_spacy.p", "rb")) as f:
    all_reviews = pickle.load(f)
print(all_reviews)
"""

