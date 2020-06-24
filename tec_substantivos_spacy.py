# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:34:17 2019

@author: yagoa
"""
import fnmatch
import pprint
import re
import numpy as np
import pickle
from enelvo import normaliser
import sys
import gensim
import string
import re
import os
import nltk
import csv
from nltk import FreqDist
from collections import Counter

import spacy
from spacy import tokens

from pathlib import Path



#pre processa texto
def pre_processing_text(text, use_normalizer=False):

    if use_normalizer:
        norm = normaliser.Normaliser()
        text = norm.normalise(text)

    text = text.lower()

    input_chars = ["\n", ".", "!", "?", "ç", " / ", " - ", "|", "ã", "õ", "á", "é", "í", "ó", "ú", "â", "ê", "î", "ô", "û", "à", "è", "ì", "ò", "ù"]
    output_chars = [" . ", " . ", " . ", " . ", "c", "/", "-", "", "a", "o", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u"]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text



#faz chamada da biblioteca spacy e atribui a uma variável
spc = spacy.load('pt_core_news_sm')

with(open("USO_GERAL.p", "rb")) as f:
    all_reviews = pickle.load(f)
#print(all_reviews)


#converte reviews lidos em string
reviews = str(all_reviews)
tratados = []

#atribuindo o texto ao modelo spacy
doc = spc(reviews)

#verifica cada review
for review in doc:

    #print(review)
    #atribuindo o texto ao modelo spacy
    

    #print(review)
    #dando split no texto
    doc.text.split()
    #print(doc)
    tagger = []
    words = []
    #realiza marcação de cada palavra
    for i, word in enumerate(doc):
        if not word.is_punct:
            if not word.is_space:
                #print(word.text, word.pos_)
                
                tagger = [(word.text, word.pos_)]#pega palavra e pos tagger
                words.append(tagger) #atribui palavra e pos tagger a lista
        if i > len(doc):
            break

    #print(words)
        tratados.append(words) #adiciona cada review a lista após tratado a lista


print(tratados)

nouns = []
for j,termo in enumerate(tratados):
    if termo[1] == 'NOUN':
        nouns.append(termo[0])

#verifica frequencia que palavras aparece
def freq(list): 
     frequencia = FreqDist(list)
     #print("\nFrequência das palavras: \n", frequencia)

freq(nouns)

#realiza poda, pegando apenas 3% das palavras mais frequentes        
def frequen(list):
     lenght = len(nouns)
     porc = (3*lenght)/100
     #print("\nQuantidade correspondente a 3%: \n", porc)


#print ("\n",len(nouns))
frequen(nouns)

fr = []
c = Counter(nouns)

fr = c.keys()
        
#print("\n",fr,"\n")

lista = []

#arquivo txt com e .p com substantivos em geral
arquivo = open('Aspectos/SUBS_Spacy.txt','w')
arquivo.write(' '.join(nouns))
arquivo.close()

arquivo = open('Aspectos/SUBS_Spacy.txt','r')
lista = arquivo.read()
#print("SUBSTANTIVOS .txt \n", lista)
arquivo.close()


with open(os.path.join("Aspectos","SUBS_Spacy.p"), "wb") as f:
        pickle.dump(nouns, f)

file = open(os.path.join("Aspectos","SUBS_Spacy.p"), "rb")
lista = pickle.load(file)
#print("SUBSTANTIVOS_Spacy .P \n", lista)    

#arquivo txt e .p com substantivos mais frequentes 
arquivo = open('Aspectos/SUBSfreq_Spacy.txt','w')
arquivo.write(' '.join(fr))
arquivo.close()

arquivo = open('Aspectos/SUBSfreq_Spacy.txt','r')
lista = arquivo.read()
#print("SUBSTANTIVOS .txt \n", lista)
arquivo.close()


with open(os.path.join("Aspectos","SUBSfreq_Spacy.p"), "wb") as f:
        pickle.dump(list(fr), f) 

file = open(os.path.join("Aspectos","SUBSfreq_Spacy.p"), "rb")
lista = pickle.load(file)
#print("SUBSTANTIVOS .P \n", lista) 

