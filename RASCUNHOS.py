"""
        while x< len(words):
            print(x,"\t",words[x])
            for Word,pos in words:
                apt = x
                anterior = words[x-1]
                if (pos == 'NN' or pos == 'NNS'):
                    print(anterior)
                    print(Word,pos)
                    #for Wrd,ps in antes:
                        #if (pos == 'JJ'):
                            #print(wrd, ps)
                            #if (pos[x-1]=='JJ'):
                             #   print(Word,pos)
            x = x+1
        
        
        #forma 1 de buscar adjetivo antes
        tam = len(words)
        for Word,pos in words:
            apont = x
            anterior = words[apont-1]
            #print("Palavras: ", Word,pos)
            #print("Anterior: ",anterior)
            #print("\n")
            if (pos == 'NN' or pos == 'NNS'):
                wrd, ps = anterior
                #print(Word,pos)
                #print("anterior: ", wrd, ps)
                #print("\n")
                if( ps == 'JJ'):
                    print(Word,pos)
                    print("anterior: ", wrd, ps)
                    #print(anterior[0],anterior[1])
                    print("\n")
            x = x+1        
            
        """

"""
            #https://www.snakify.org/pt/lessons/two_dimensional_lists_arrays/

        tam = len(words)
        for i,termo in enumerate(words):
            apont = i
            posicao = tam-i+1
            anterior=words[apont-1]
            posterior = words[posicao]
            print("termo: ",termo[i])
            print("anterior: ",anterior)
            print("posterior: ",posterior)
            print("\n")
        """




# -*- coding: utf-8 -*-

"""
from pathlib import Path
import pickle


#lendo arquivo com informações de polaridade
sentilexpt = open("SentiLex-PT01/SentiLex-lem-PT01.txt",'r',encoding="utf8")
sentilexpt = sentilexpt.readlines()

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


#iterando sobre todas as linhas do arquivo para obter as palavras e polaridade de cada uma
dic_palavra_polaridade = {}
for i in sentilexpt:
    pos_ponto = i.find('.')
    palavra = pre_processing_text((i[:pos_ponto]))
    pol_pos = i.find('POL')
    polaridade = (i[pol_pos+4:pol_pos+6]).replace(';','')
    dic_palavra_polaridade[palavra] = polaridade


print(dic_palavra_polaridade)

print(sentilexpt)


import gensim
import numpy as np
import os
import pickle
import fnmatch
from enelvo import normaliser

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
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/testando"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review = f.read()
        review = pre_processing_text(review, use_normalizer=True)
        all_reviews.append(review)
with open("USO_GERAL.p", "wb") as f:
    pickle.dump(all_reviews, f)

"""

import gensim
import numpy as np
import os
import pickle
import fnmatch
from enelvo import normaliser


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



all_reviews_n = []
all_reviews_p = []
all_reviews_t = []

for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/testando/negativo"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review_n = f.read()
        review_n = [pre_processing_text(review_n, use_normalizer=True), '-1']
        all_reviews_n.append(review_n)

for dirpath, _, files in os.walk("./Corpus Buscape/treinamento//testando/positivo"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review_p = f.read()
        review_p = [pre_processing_text(review_p, use_normalizer=True), '1']
        all_reviews_p.append(review_p)


all_reviews_t = all_reviews_n + all_reviews_p
print(all_reviews_t)

with open("USO_GERAL1.p", "wb") as f:
    pickle.dump(all_reviews_t, f)

"""


import gensim
import numpy as np
import os
import pickle
import fnmatch
from enelvo import normaliser
from unidecode import unidecode
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.corpus import stopwords
import re
from unicodedata import normalize

def pre_processing_text(text, use_normalizer=False):

    if use_normalizer:
        norm = normaliser.Normaliser()
        text = norm.normalise(text)

    text = text.lower()

    input_chars = ["\n", ".", "!", "?", "ç", " / ", " - ", "|", "ã", "õ", "á", "é", "í", "ó", "ú", "â", "ê", "î", "ô", "û", "à", "è", "ì", "ò", "ù", "@","#", "$", "%", "&", "*", "(", ")", "[", "]", "{", "}", ";", ":", "<", ">", "=", "_", "+"]
    output_chars = [" . ", " . ", " . ", " . ", "c", "/", "-", "", "a", "o", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u","","","","","","","","","","","","","","","","","","",""]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text





def removerCaracteresEspeciais(text):
    
    #Método para remover caracteres especiais do texto
    
    return normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')


all_tokenized_reviews = []
print("Processed_Reviews.p couldn't be found. All reviews will be loaded from txt files, this will take a fell minutes")
all_reviews = []
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review = f.read()
        review = pre_processing_text(review, use_normalizer=True)

        rev = removerCaracteresEspeciais(review)

        #rev = re.sub('[#$%^&*()[]{};:,<>\`~=_+]', ' ', review)
        
        all_reviews.append(rev)
        
arquivo = open('Aspectos/reviews_for_Palavras.txt','w')
arquivo.write('\n'.join(all_reviews))
arquivo.close()


import gensim
import numpy as np
import os
import pickle
import fnmatch
from enelvo import normaliser


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



all_reviews_n = []
all_reviews_p = []
all_reviews_t = []

for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/testando"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review_n = f.read()
        review_n = [pre_processing_text(review_n, use_normalizer=True), '-1']
        all_reviews_n.append(review_n)


all_reviews_t = all_reviews_n


with open("USO_GERAL1.p", "wb") as f:
    pickle.dump(all_reviews_t, f)





# -*- coding: utf-8 -*-

Created on Wed Dec  4 10:54:31 2019

@author: yagoa

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
tratados = []
with(open("USO_GERAL.p", "rb")) as f:
    all_reviews = pickle.load(f)
#print(all_reviews)

#reviews = str(all_reviews)
for review in all_reviews:
    review= str(review)
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
                tagger = [word.text, word.pos_]
                words.append(tagger)
        if i > len(doc):
            break

    print(words)
    tratados.append(words)


"""


