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
import os
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('pos_tag')
stop_words = nltk.corpus.stopwords.words('portuguese')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from enelvo import normaliser
from unidecode import unidecode
from nltk import FreqDist
from collections import Counter

from pathlib import Path


with(open(os.path.join("USO_GERAL.p"), "rb")) as f:
    all_reviews = pickle.load(f)
#print(all_reviews)


    

#cria objeto normalizador com atributos padrão
norm = normaliser.Normaliser()
#normaliza a sentença
norm_sentence = norm.normalise(conteudo)
#coloca toda sentença em minúsculo
norm_sentence = norm_sentence.lower()
#possibilita mudar o atributo que desejar
norm.capitalize_inis = True
print("TEXTO NORMALIZADO: \n")

print(norm_sentence,"\n")

print("TEXTO SEM ACENTUAÇÃO: \n")

print(unidecode(norm_sentence))

temp = unidecode(norm_sentence)

#divide as palavras em uma lista(split)
tokens = word_tokenize(temp)

#remove pontuação de cada palavra
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]

#remove qualquer outro caracter que não seja do alfabeto
words = [word for word in stripped if word.isalpha()]

print("\nSEM PONTUAÇÃO:\n",words)

stop_words = set(stopwords.words('portuguese'))
words = [w for w in words if not w in stop_words]

print("\nSEM STOP WORDS:\n",words)

words = nltk.pos_tag(words)
print("\nPOS_TAG:\n", words)

arquivo = open('tagger_nltk.txt','w')
arquivo.write(' '.join(map(str, words)))
arquivo.close()

with open(os.path.join("tagger_nltk.p"), "wb") as f:
        pickle.dump(list(words), f)
        
with(open("tagger_nltk.p", "rb")) as f:
    all_reviews = pickle.load(f)
print(all_reviews)

