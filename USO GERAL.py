# -*- coding: utf-8 -*-


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


    input_chars = ["\n"," / ", " - ", "|","@","#", "$", "%", "&", "*", "(", ")", "[", "]", "{", "}", ";", ":", "<", ">", "=", "_", "+",]
    output_chars = [".","/", "-", "","","","","","","","","","","","","","","","","","","","",]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text





def removerCaracteresEspeciais(text):
    
    #Método para remover caracteres especiais do texto
    
    return normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')


all_tokenized_reviews = []
print("reviews_for_Palavras.txt couldn't be found. All reviews will be loaded from txt files, this will take a fell minutes")
all_reviews = []
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/testando/negativo"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review = f.read()
        review = pre_processing_text(review)

        #rev = removerCaracteresEspeciais(review)

        #rev = re.sub('[#$%^&*()[]{};:,<>\`~=_+]', ' ', review)
        
        all_reviews.append(review)
        
arquivo = open('rt_polarity_neg.txt','w')
arquivo.write('\n'.join(all_reviews))
arquivo.close()

"""
import pprint
import pickle
import nltk
import string

import gensim
import fnmatch
import enelvo
import re

from enelvo import normaliser
from collections import Counter

import spacy
import os
from spacy import tokens
from pathlib import Path



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

def lexico_sentimento_SentiLex(review):
  

    #lendo arquivo com informações de polaridade
    sentilexpt = open("SentiLex-PT01/SentiLex-lem-PT01.txt",'r',encoding="utf8")
    dic_palavra_polaridade = {}
    for i in sentilexpt.readlines():
        pos_ponto = i.find('.')
        palavra = pre_processing_text((i[:pos_ponto]))
        pol_pos = i.find('POL')
        polaridade = (i[pol_pos+4:pol_pos+6]).replace(';','')
        dic_palavra_polaridade[palavra] = polaridade

    l_sentimento= []
    word_sentimento = []
    w = []

    for word in review:
        w = [word,int(dic_palavra_polaridade.get(word,0))]
        word_sentimento.append(w)

    #retorna lista palavra e a polaridade de cada palavra
    return (word_sentimento)

def Sentilex():

    sentilexpt = open("SentiLex-PT01/SentiLex-lem-PT01.txt",'r',encoding="utf8")
    dic_palavra_polaridade = {}
    for i in sentilexpt.readlines():
        pos_ponto = i.find('.')
        palavra = pre_processing_text((i[:pos_ponto]))
        pol_pos = i.find('POL')
        polaridade = (i[pol_pos+4:pol_pos+6]).replace(';','')
        dic_palavra_polaridade[palavra] = polaridade

    #retorna dicionário do SentiLex
    return(dic_palavra_polaridade)

        

def tec_posicao_adjetivo_spacy(all_reviews):

    all_tokenized_reviews = []
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    
    spc = spacy.load('pt_core_news_sm')
    
    with open(os.path.join("USO_GERAL1.p"), "rb") as file:  #Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
    result_review = []

    spc = spacy.load('pt_core_news_sm')
    tratados = []

    
    
    sent_words = Sentilex()
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
                    plvra = palavra.text
                    lista.append(plvra)

        frase_polarity = lexico_sentimento_SentiLex(lista)
        print("FRASE_POLARITY:\t",frase_polarity)
        print("\n")

        #print(wd)
        #print("\n")
        
        tagger = []
        wd = []
        for i,word in enumerate(words):
            #print(i,word)
            
            if not word.is_punct:
                if not word.is_space:
                    #print(word.text, word.pos_)
                    tagger = [word.text, word.pos_]
                    wd.append(tagger)
            if i > len(words):
                break

        
        
        for j,termo in enumerate(wd):
            apont = j
                
            anterior=wd[j-1]
                
            #print("ANTERIOR: ",anterior)
            #print("PALAVRA \t",j,termo)
                
            if(termo[1] == 'NOUN'):
                #print("\n****** ENTROU \t 1 ******\n", termo)
                x=j+2
                    
                posterior = wd[j+1:x]
                #print("POSTERIOR", posterior)
                #print("Verificou substantivo\n")
                        
                for wrd,ps in posterior:
                    post = wrd
                        
                #print("verificando se: ",anterior[0]," é negação")
                    
                if anterior[0] in negacao:
                    #print("\n****** ENTROU \t 2 ******\n", anterior)
                    #print("verificou negação")
                    palavra = termo[0]
                    for l,item in enumerate(frase_polarity):
                        ap = l
                        y = l+2
                                
                        poeio=frase_polarity[l+1:y]
                                
                        for w,p in poeio:
                            pt = w
                                   
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                    
                    
                if(anterior[1]=='ADJ'):
                    ant = anterior[0]
                    #print("\n****** ENTROU \t 3 ******\n", anterior)
                    #print("Entrou Adjetivo antes\n")
                    #print(ant)
                    #print(anterior[1])
                    palavra = termo[0]
                        
                        
                        
                    for k,item in enumerate(frase_polarity):
                        #print("\n****** ENTROU \t 4 ******\n", item)
                        ap = k
                            
                        z = k+2
                            
                        poeio=frase_polarity[k+1:z]
                            
                        for w,p in poeio:
                            pt = w

                                
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("\n****** ENTROU \t 5 ******\n", item)

                                 
                        if palavra == item[0]:
                            item[1] = polaridade_adj
                            #print("\n****** ENTROU \t 6 ******\n", item)
                                          
            
                for wrd,ps in posterior:
                    if(ps == 'ADJ'):
                        #print("Entrou adjetivo depois",)                        
                        palavra = termo[0]
                        ant = anterior[0]
                            
                        for k,item in enumerate(frase_polarity):
                            ap = k
                                
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                                
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                    
                                    
                            #print("VERIFICANDO DENTRO DA LISTA COM POLARIDADE")
                                if wrd == item[0]:
                                    polaridade_adj_pos = item[1]
                                    #print("entrou e recolheu polaridade1")
                                    
                                if palavra == item[0]:
                                    item[1] = polaridade_adj_pos
                                    #print("entrou e atribuiu polaridade")
                        
                           
              
        print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
        polaridade_rev = 0
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
        
        print("\n",polaridade_rev,"\n")
        
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)
            
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)
            
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)

        print("REVIEW Nº:\t",cont)
            
    acertos = 0 
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i]:
            acertos += 1
        else:
            print("")
    
                
    print("TOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    acuracia = acertos/(len(all_reviews))*100
    print("\n\n\n\n\nacuracia:\t",acuracia,"%")

    

all_reviews = []

with open(os.path.join("USO_GERAL.p"), "rb") as file:
        all_reviews = pickle.load(file)


tec_posicao_adjetivo_spacy(all_reviews)


with open(os.path.join("reviews_for_Palavras.p"), "rb") as file:
        all_reviews = pickle.load(file)


arquivo = open('reviews_for_Palavras.txt','w')
arquivo.writelines(str(all_reviews))
arquivo.close()
"""
