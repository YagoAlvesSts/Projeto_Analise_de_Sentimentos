# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:54:31 2019

@author: yagoa
"""
import pprint
import pickle
import nltk
import string
import os
import gensim
import fnmatch
import enelvo
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('pos_tag')
stop_words = nltk.corpus.stopwords.words('portuguese')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from unidecode import unidecode
from nltk import FreqDist
from enelvo import normaliser
from collections import Counter
from pathlib import Path

import spacy
from spacy import tokens



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

        

def tec_posicao_adjetivo_nltk(all_reviews):

    all_tokenized_reviews = []
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    

    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file:
        polarity_reviews = pickle.load(file)
    result_review = []

    #CHAMA DICIONARIO DO SENTILEX Processed_Reviews_polarity
    #print("DICIONARIO SENTILEX: \n",Sentilex())
    sent_words = Sentilex()
    cont = 0
    
    for review in all_reviews:
        
        norm = normaliser.Normaliser()
        #normaliza a sentença
        norm_sentence = norm.normalise(review)
        #coloca toda sentença em minúsculo
        norm_sentence = norm_sentence.lower()
        #possibilita mudar o atributo que desejar
        norm.capitalize_inis = True

        temp = unidecode(norm_sentence)

        #divide as palavras em uma lista(split)
        tokens = word_tokenize(temp)

        #remove pontuação de cada palavra
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        #remove qualquer outro caracter que não seja do alfabeto
        words = [word for word in stripped if word.isalpha()]


        stop_words = set(stopwords.words('portuguese'))
        words = [w for w in words if not w in stop_words]


        #CHAMANDO FUNÇÃO DE ATRIBUIR POLARIDADE
        frase_polarity = lexico_sentimento_SentiLex(words)
        
        
        #dic_frase_polarity = {}
        
        #for item in frase_polarity:
         #   dic_frase_polarity[item[0]] = item[1]
                        
        #print("\nDicionário:\n",dic_frase_polarity,"\n")

        
        
        #print("\nPALAVRAS COM POLARIDADE:\n",frase_polarity)


        words = nltk.pos_tag(words)

        
        #print("\n",words)

        
        #forma 2 de buscar adjetivo antes
        
        for i,termo in enumerate(words):
            apont = i
            anterior=words[apont-1]
            
            if(termo[1] == 'NN' or termo[1] == 'NNS'):
                x=i+2
                posterior = words[i+1:x]
                #print("Verificou substantivo\n")
                
                for wrd,ps in posterior:
                    post = wrd
                
                if anterior[0] in negacao:
                    palavra = termo[0]
                    for j,item in enumerate(frase_polarity):
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        
                        for wd,p in poeio:
                            pt = wd
                           
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                elif(anterior[1]=='JJ'):
                    #print("Entrou Adjetivo antes\n")
                    palavra = termo[0]
                    ant = anterior[0]
                    #print(ant)
                    
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wd,p in poeio:
                            pt = wd

                            
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            
                        if palavra == item[0] and post == pt:
                            item[1] = polaridade_adj
                            
                            
                        
                
                for wrd,ps in posterior:
                    if(ps == 'JJ'):
                                                
                        palavra = termo[0]
                        ant = anterior[0]
                        
                        for k,item in enumerate(frase_polarity):
                            ap = k
                            
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                            
                            for a,b in post_polar:
                                    polaridade_adj_pos = b
                                
                                
                            
                            if wrd == item[0]:
                                polaridade_adj_pos = item[1]
                                
                            if palavra == item[0] and ant == antes:
                                item[1] = polaridade_adj_pos
                        
                           
                
        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
        polaridade_rev = 0
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
        
        print("\n",polaridade_rev)
        
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
    
                
    print("\nTOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    acuracia = acertos/(len(all_reviews))*100
    print("\n\n\n\n\nacuracia:\t",acuracia,"%")

            
def tec_posicao_adjetivo_spacy(all_reviews):
    all_tokenized_reviews = []
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    
    spc = spacy.load('pt_core_news_sm')
    
    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file:  #Processed_Reviews_polarity
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
        #print("FRASE_POLARITY:\t",frase_polarity)
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
                        
                           
              
        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
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
"""
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/lexico"):
    for filename in fnmatch.filter(files, '*.txt'):
            f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
            review = f.read()
            print(review)
            review = pre_processing_text(review, use_normalizer=True)
            all_reviews.append(review)
    with open("tec_linha_de_base.p", "wb") as f:
        pickle.dump(all_reviews, f) Processed_Reviews 
"""
with open(os.path.join("Processed_Reviews.p"), "rb") as file:
        all_reviews = pickle.load(file)


tec_posicao_adjetivo_spacy(all_reviews)


