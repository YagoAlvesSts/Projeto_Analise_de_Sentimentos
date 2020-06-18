# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:34:17 2019

@author: yagoa
"""
import fnmatch
import re
import numpy as np
import pickle
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



all_reviews = []
#realiza leitura de todos os reviews de treinamento
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento"):
        for filename in fnmatch.filter(files, '*.txt'):
            f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
            conteudo = f.read()
            all_reviews.append(conteudo)
            
#print(all_reviews)


# Criando e escrevendo em arquivos de texto (modo 'w').
arquivo = open('matriz01.txt','w')
arquivo.writelines(all_reviews)
arquivo.close()


# Lendo o arquivo criado:
arquivo = open('matriz01.txt','r')
conteudo=arquivo.read()

#all_tokenized_reviews = []
    
#faz chamada da biblioteca spacy e atribui a uma variável
spc = spacy.load('pt_core_news_sm')
tratados = []
cont = 0
#verifica cada review
conteudo = spc(conteudo)

for i,word in enumerate(conteudo):

    #print(word)
    #dando split no texto
    #words.text.split()
    lista = []

    
    if not word.is_punct:
        if not word.is_space:
            tagger = [pre_processing_text(word.text), word.pos_] #pega palavra e pos tagger
            tratados.append(tagger) #atribui palavra e pos tagger a lista
                
print(tratados)
nouns = []
for j,termo in enumerate(tratados):
    if termo[1] == 'NOUN':
        nouns.append(termo[0])

#verifica frequencia que palavras aparece
def freq(list): 
     frequencia = FreqDist(list)
     print("\nFrequência das palavras: \n", frequencia)

freq(nouns)

#realiza poda, pegando apenas 3% das palavras mais frequentes        
def frequen(list):
     lenght = len(nouns)
     porc = (3*lenght)/100
     print("\nQuantidade correspondente a 3%: \n", porc)


print ("\n",len(nouns))
frequen(nouns)

fr = []
c = Counter(nouns)

fr = c.keys()
        
print("\n",fr,"\n")

lista = []

#arquivo txt com e .p com substantivos em geral
arquivo = open('Aspectos/SUBS_Spacy.txt','w')
arquivo.write(' '.join(nouns))
arquivo.close()

arquivo = open('Aspectos/SUBS_Spacy.txt','r')
lista = arquivo.read()
print("SUBSTANTIVOS .txt \n", lista)
arquivo.close()


with open(os.path.join("Aspectos","SUBS_Spacy.p"), "wb") as f:
        pickle.dump(nouns, f)

file = open(os.path.join("Aspectos","SUBS_Spacy.p"), "rb")
lista = pickle.load(file)
print("SUBSTANTIVOS_Spacy .P \n", lista)    

#arquivo txt e .p com substantivos mais frequentes 
arquivo = open('Aspectos/SUBSfreq_Spacy.txt','w')
arquivo.write(' '.join(fr))
arquivo.close()

arquivo = open('Aspectos/SUBSfreq_Spacy.txt','r')
lista = arquivo.read()
print("SUBSTANTIVOS .txt \n", lista)
arquivo.close()


with open(os.path.join("Aspectos","SUBSfreq_Spacy.p"), "wb") as f:
        pickle.dump(list(fr), f) 

file = open(os.path.join("Aspectos","SUBSfreq_Spacy.p"), "rb")
lista = pickle.load(file)
print("SUBSTANTIVOS .P \n", lista) 

