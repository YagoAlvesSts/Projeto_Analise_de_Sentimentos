# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:54:31 2019

@author: yagoa
"""
#BLIBLIOTECAS GERAIS
import pprint
import pickle
import nltk
import string
import os
import gensim
import fnmatch
import enelvo
import re
from enelvo import normaliser
from ftfy import fix_encoding
import subprocess
from collections import Counter

#BIBLIOTECA PARA LER SENTIWORDNET-PT-BR
import pandas as pd

#BIBLIOTECAS DO NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('pos_tag')
stop_words = nltk.corpus.stopwords.words('portuguese')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from unidecode import unidecode
from nltk import FreqDist

#BIBLIOTECAS PARA ABRIR PASTAS
from collections import Counter
from pathlib import Path

#BIBLIOTECAS PARA UTILIZAR O SPACY
import spacy
from spacy import tokens


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
"""

SentiWordNet = []
#busca arquivo com o pandas (pq está estruturado em colunas, tipo tabela)
df = pd.read_csv('léxico/SentiWordNet_PT/SentiWord Pt-BR v1.0b.txt', delimiter="\t", header=None, names=["ID","PosScore", "NegScore", "Termo"])
#print(df.values)
#pega valores do arquivo e gera lista com listas dentro contendo (código, posScore, negScore, e o termo em si)
SentiWordNet = df.values


#Como funciona após adicionar valores das colunas em uma lista
scorepos = 0.0
scoreneg = 0.0
pol = 0
#print(SentiWordNet)
for i,termo in enumerate(SentiWordNet):
    if termo[3] == 'miserável':
        scorepos = scorepos + float(termo[1])
        scoreneg = scoreneg + float(termo[2])
        
        if(scorepos > scoreneg):
            pol = '1'
        if(scorepos > scoreneg):
            pol = '-1'
        else:
            pol = '0'
        #existe mais de uma polaridade no léxico para uma palavra (geralmente é igual)
        print("termo:",termo[3],"\tposscore: ",termo[1],"\tnegscore: ",termo[2])
        print(pol)


#realiza tratamento do texto
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

def lexico_sentimento_SentWordNetPT(review):
   
    #lista que será adicionado review pós adiconar polaridade de cada palavra
    word_sentimento = []
    
    pol = 0
    word_pol = []
    #atribui polaridade a cada palavra do review
    for word in review:
        word = pre_processing_text(word) #trata o texto
        #chama função de atribuir polaridade a palavra
        polaridade = atribui_polaridade_sentiwordnet(word)
        
        if(polaridade !=  None ):#se polaridade existir no léxico, atribui ao score
            #print()
            scorepos= polaridade[0]
            scoreneg=polaridade[1]
            if(scorepos > scoreneg):
                pol = '1'
            elif(scorepos < scoreneg):
                pol = '-1'
            else:
                pol = '0'
                
            word_pol = [word,pol]
            #adiciona palavra e polaridade a lista
            word_sentimento.append(word_pol)
        if(polaridade == None): #se palavra não existe no léxico, atribui 0
            word_pol = [word,'0']
            word_sentimento.append(word_pol)
        
    return (word_sentimento) #retorna review com polaridades


def atribui_polaridade_sentiwordnet(word):
    #lista que será adicionado valores do léxico
    SentiWordNet = []
    #lê léxico com o pandas
    df = pd.read_csv('léxico/SentiWordNet_PT/SentiWord Pt-BR v1.0b.txt', delimiter="\t", header=None, names=["ID","PosScore", "NegScore", "Termo"])
    #print(df.values)
    SentiWordNet = df.values #pega valores do léxico de polaridades
    scorepos = 0.0
    scoreneg = 0.0
    neg = []
    pos = []
    cont = 0
    #busca palavra no léxico
    for i,termo in enumerate(SentiWordNet):
        trm = pre_processing_text(termo[3]) #trata a palavra a ser buscada
        if trm == word: #compara palavra do review com o léxico
            cont += 1
            print(termo[3],trm,word)
            #aqui o ideal seria somar, mas atualmente ele só pega polaridade da ultima palavra encontrada
            neg.append(float(termo[1]))
            pos.append(float(termo[2]))


            
            #scorepos = scorepos + float(termo[1])#soma polaridades positivas
            #scoreneg = scoreneg + float(termo[2])#soma polaridades negativas
            
            print("termo:",termo[3],"\tposscore: ",pos,"\tnegscore: ",neg)
        for i,score in neg:
            scoreneg += score[i]

        for j,score in pos:
            scorepos += score[j]

            
        return (scorepos,scoreneg) #retorna score de polaridades

def tec_posicao_adjetivo_spacy(all_reviews):
    all_tokenized_reviews = []
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    
    spc = spacy.load('pt_core_news_sm')
    
    with open(os.path.join("USO_GERAL1.p"), "rb") as file:  #Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
    result_review = []

    spc = spacy.load('pt_core_news_sm')
    tratados = []

    cont = 0
    
    for review in all_reviews:

        review= str(review)
        #atribuindo o texto ao modelo spacy
        words = spc(review)


        #dando split no texto
        words.text.split()
        lista = []
        
        for i,palavra in enumerate(words):
            if not palavra.is_punct:
                if not palavra.is_space:
                    if not palavra.is_stop:
                        plvra = palavra.text
                        lista.append(plvra)

        frase_polarity = lexico_sentimento_SentWordNetPT(lista)
        #print("FRASE_POLARITY:\t",frase_polarity)
        print("polaridades com SentWordNetPT:\n",frase_polarity)
        

all_reviews = []

with open(os.path.join("USO_GERAL.p"), "rb") as file:
        all_reviews = pickle.load(file)


tec_posicao_adjetivo_spacy(all_reviews)

with open(os.path.join("léxico/","SentWordNet_PTBR.p"), "rb") as f:
    sent_words = pickle.load(f)
with open(os.path.join("léxico/","SentWordNet_PTBR_polarity.p"), "rb") as f:
    sent_words_polarity = pickle.load(f)

print("sent_words: \n",sent_words)
print("sent_words_polarity: \n",sent_words_polarity)
"""


    
    
sent_words=[]
sent_w = []
df = pd.read_csv('léxico/SentiWordNet_PT/SentiWord Pt-BR v1.0b.txt', delimiter="\t", header=None) #names=["ID","PosScore", "NegScore", "Termo"]
#print(df.values)
SentiWordNet = df.values #pega valores do léxico de polaridades

pedaco = []
SentiWordNet = SentiWordNet[1:]


        
sent_words = SentiWordNet     #SentiWordNet
sent_w = SentiWordNet     #SentiWordNet

extraidos = []
palavrasSentimento = []
sent_words_polarity = {}

#print(SentiWordNet)
#print(sent_words)
#print(sent_w)

scorepos = 0.0
scoreneg = 0.0
neg = []
pos = []
save = True


for i,termo in enumerate(sent_words):
    trm = pre_processing_text(termo[3])
    if trm in extraidos:
        print("Já existe!")
    else:
        
        for j,item in enumerate(sent_w):
            
            it = pre_processing_text(item[3])
            if it == trm:
                print(item)
                
                neg.append(float(item[2]))
                pos.append(float(item[1]))
                
        scoreneg = sum(k for k in neg)
        scorepos = sum(l for l in pos)
        
        print("Negativo: ",scoreneg,"Positivo: ",scorepos)
        
        pol = 0
        
        if(scorepos > scoreneg):
            pol = '1'
            
        
            
        if(scorepos < scoreneg):
            pol = '-1'
            
            
        if (scorepos == scoreneg):
            pol = '0'
            
        pos.clear()
        neg.clear()
        
        word_pol = [trm,pol]
        
        extraidos.append(trm)
        palavrasSentimento.append(word_pol)
        sent_words_polarity[trm] = pol

if save:#salva léxico
    with open(os.path.join("léxico/","SentWordNet_PTBR.p"), "wb") as f:
        pickle.dump(sent_words, f)
    with open(os.path.join("léxico/","SentWordNet_PTBR_polarity.p"), "wb") as f:
        pickle.dump(sent_words_polarity, f)
        
                





































