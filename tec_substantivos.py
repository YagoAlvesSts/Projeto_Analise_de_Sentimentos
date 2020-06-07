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


"""
#Abrindo vários arquivos
path = Path("C:/Users/yagoa/Desktop/Faculdade/PP/ultimos/Corpus Buscape/treinamento")
letters_files = path.glob('*.txt')
letters = [letter.read_text(encoding = 'utf-8') for letter in sorted(letters_files)]

print(letters)
"""
all_reviews = []
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento"):
        for filename in fnmatch.filter(files, '*.txt'):
            f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
            conteudo = f.read()
            all_reviews.append(conteudo)
            
print(all_reviews)


# Criando e escrevendo em arquivos de texto (modo 'w').
arquivo = open('matriz01.txt','w')
arquivo.writelines(all_reviews)
arquivo.close()


# Lendo o arquivo criado:
arquivo = open('matriz01.txt','r')
conteudo=arquivo.read()

print(conteudo)

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

print("\nPOS_TAG:\n",nltk.pos_tag(words))

#print([Word for (Word, pos) in nltk.pos_tag(nltk.word_tokenize(words))
 #    if str(pos) == 'NN' or str(pos) == 'NNP' or str(pos) == 'NNS' or str(pos) == 'NNPS'])
nouns = [] #empty to array to hold all nouns
#for word in words:
for Word,pos in nltk.pos_tag(words):
           if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.append(Word)

print("\nSUBSTANTIVOS:\n",nouns)


def freq(list):
     frequencia = FreqDist(list)
     print("\nFrequência das palavras: \n", frequencia)

freq(nouns)

        
def frequen(list):
     lenght = len(nouns)
     porc = (3*lenght)/100
     print("\nQuantidade correspondente a 3%: \n", porc)


print (len(nouns))
frequen(nouns)

fr = []
c = Counter(nouns)

fr = c.keys()
        
print(fr)

lista = []

#arquivo txt com e .p com substantivos em geral
arquivo = open('Aspectos/SUBS.txt','w')
arquivo.write(' '.join(nouns))
arquivo.close()

arquivo = open('Aspectos/SUBS.txt','r')
lista = arquivo.read()
print("SUBSTANTIVOS .txt", lista)
arquivo.close()


with open(os.path.join("Aspectos","SUBS.p"), "wb") as f:
        pickle.dump(nouns, f)

file = open(os.path.join("Aspectos","SUBS.p"), "rb")
lista = pickle.load(file)
print("SUBSTANTIVOS .P", lista)    

#arquivo txt e .p com substantivos mais frequentes 
arquivo = open('Aspectos/SUBSfreq.txt','w')
arquivo.write(' '.join(fr))
arquivo.close()

arquivo = open('Aspectos/SUBSfreq.txt','r')
lista = arquivo.read()
print("SUBSTANTIVOS .txt", lista)
arquivo.close()


with open(os.path.join("Aspectos","SUBSfreq.p"), "wb") as f:
        pickle.dump(list(fr), f) 

file = open(os.path.join("Aspectos","SUBSfreq.p"), "rb")
lista = pickle.load(file)
print("SUBSTANTIVOS .P", lista) 
