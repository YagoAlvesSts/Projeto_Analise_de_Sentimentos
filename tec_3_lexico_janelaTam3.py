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

cobertura = 0
precisao = 0
mediaf = 0
acuracia = 0
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

#atribui polaridade com Sentilex
def lexico_sentimento_SentiLex(review,save=True):
  
    try:#verifica se léxico já foi lido uma vez e salvo
        with open(os.path.join("léxico/","SentiLex_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/","SentiLex_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        #lendo arquivo com informações de polaridade
        sentilexpt = open("SentiLex-PT02/SentiLex-flex-PT02.txt",'r',encoding="utf8")
        sent_words_polarity = {}

        text = sentilexpt.readlines()
        sent_words = []
        for line in text:
                line = line.split(',')
                word = line[0]
                word = pre_processing_text(word)
                #word = N.unidecode(word) #tira acentuação
                try:#busca polaridade
                    polarity = line[1].split('N0=')[1].split(';')[0]
                except:
                    polarity = line[1].split('N1=')[1].split(';')[0]
                sent_words.append(word)
                #cria dicionário com polaridades
                sent_words_polarity[word] = polarity
    if save:#salva léxico
        with open(os.path.join("léxico/","SentiLex_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/","SentiLex_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)
            
    l_sentimento= []
    word_sentimento = []
    w = []
    #busca polaridade de cada palavra do review
    for word in review:
        w = [word,int(sent_words_polarity.get(word,0))]
        word_sentimento.append(w)#adiciona palavra com polaridade em lista

    #retorna lista palavra e a polaridade de cada palavra
    return (word_sentimento)

def Sentilex():
    #lendo arquivo com informações de polaridade
    sentilexpt = open("SentiLex-PT02/SentiLex-flex-PT02.txt",'r',encoding="utf8")
    dic_palavra_polaridade = {}
    text = sentilexpt.readlines()
    sent_words = []
    for line in text:
            line = line.split(',')
            word = line[0]
            word = pre_processing_text(word)
            #word = N.unidecode(word) #tira acentuação
            try:#busca polaridade
                polarity = line[1].split('N0=')[1].split(';')[0]
            except:
                polarity = line[1].split('N1=')[1].split(';')[0]
            sent_words.append(word)
            #cria dicionário com polaridades
            dic_palavra_polaridade[word] = polarity

    #retorna dicionário do SentiLex
    return(dic_palavra_polaridade)


def lexico_sentimento_SentWordNetPT(review,save=True):


    with open(os.path.join("léxico/","SentWordNet_PTBR.p"), "rb") as f:
        sent_words = pickle.load(f)
    with open(os.path.join("léxico/","SentWordNet_PTBR_polarity.p"), "rb") as f:
        sent_words_polarity = pickle.load(f)
    
            
    word_sentimento = []
    
    #busca cada palavra do review
    for word in review:
        wrd = [word,int(sent_words_polarity.get(word,0))] #pega palavra com polaridade no dicionario
        word_sentimento.append(wrd)
       
    return (word_sentimento) #retorna review com polaridades


def atribui_polaridade_sentiwordnet(word):
    
    sent_words_polarity = {}
    #lista que será adicionado valores do léxico
    SentiWordNet = []
    sent_words =[]
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
            #print(termo[3],trm,word)
            #aqui o ideal seria somar, mas atualmente ele só pega polaridade da ultima palavra encontrada
                
            neg.append(float(termo[1]))
            pos.append(float(termo[2]))


                
            #scorepos = scorepos + float(termo[1])#soma polaridades positivas
            #scoreneg = scoreneg + float(termo[2])#soma polaridades negativas
                
            #print("\tposscore: ",pos,"\tnegscore: ",neg)
            
                
        scoreneg = sum(i for i in neg)

        scorepos = sum(j for j in pos)
        
            
    #print(scorepos,scoreneg)    
    return (scorepos,scoreneg) #retorna score de polaridades

def lexico_sentimento_LIWC(review, save=True):
    
    try: #verifica se léxico já foi lido uma vez e salvo
        with open(os.path.join("léxico/","LIWC_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/","LIWC_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        sent_words = []
        sent_words_polarity = {}
        #busca arquivo com léxico LIWC
        with open("léxico/liwc.txt", encoding="utf8") as f:
            text = f.readlines()
            for line in text: 
                word = line.split()[0]
                #print("Line",line)
                
                word = pre_processing_text(word)
                #word = line.split()[0]
                #BUSCA POLARIDADE
                if "126" in line:
                    # Positive sentiment word
                    sent_words.append(word)
                    sent_words_polarity[word] = "1" #cria dicionário de léxico com polaridade
                if "127" in line:
                    sent_words.append(word)
                    sent_words_polarity[word] = "-1" #cria dicionário de léxico com polaridade
                else:
                    sent_words.append(word)
                    sent_words_polarity[word] = "0" #cria dicionário de léxico com polaridade
        # Remove duplicated words
        sent_words = list(set(sent_words))
    if save:
        with open(os.path.join("léxico/","LIWC_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/","LIWC_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)

    l_sentimento= []
    word_sentimento = []
    w = []

    #busca polaridade de cada palavra do review
    for termo in review: 
        w = [termo,int(sent_words_polarity.get(termo,0))]
        word_sentimento.append(w)

    return (word_sentimento) #retorna lista com palavras e polaridades

#Atribui polaridade com léxico OpLexicon
def lexico_sentimento_OpLexicon(review):
    try: #verifica se léxico já foi lido uma vez e salvo
        with open(os.path.join("léxico/","OpLexicon_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/","OpLexicon_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        print("\nProcessing Lexico")
        sent_words = []
        sent_words_polarity = {}
        #busca arquivo com léxico OpLexicon
        f = open("léxico/lexico_v3.0.txt",'r',encoding="utf8")
        text = f.readlines()
        #BUSCA POLARIDADE
        for line in text:
            line = line.split(',')
            word = line[0]
            word = pre_processing_text(word)
            #word = N.unidecode(word) #tira acentuação
            polarity = line[2]
            sent_words.append(word)
            sent_words_polarity[word] = polarity #cria dicionário com polaridade
        
    with open(os.path.join("léxico/","OpLexicon_sent_words.p"), "wb") as f:
        pickle.dump(sent_words, f)
    with open(os.path.join("léxico/","OpLexicon_sent_words_polarity.p"), "wb") as f:
        pickle.dump(sent_words_polarity, f)

    l_sentimento= []
    word_sentimento = []
    w = []
    #busca polaridade de cada palavra do review
    for termo in review:
        w = [termo,int(sent_words_polarity.get(termo,0))]
        word_sentimento.append(w)

    return (word_sentimento) #retorna lista com palavras e polaridades

#função que concatena léxicos de polaridade
def concatenar(lexico_1, lexico_2, lexico_3, review, save=True):

    try: #busca se léxicos já foram concatenados anteriormente
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:#chama cada léxico separadamente para depois concatenar
        lexico_sentimento_OpLexicon(review)
        lexico_sentimento_SentiLex(review)
        lexico_sentimento_LIWC(review)

        #SALVA LÉXICOS QUE FORAM ABERTOS ATRAVÉS DE CHAMADA DE FUNÇÃO
        f = open(os.path.join("léxico/",lexico_1+"_sent_words.p"), "rb")
        sent_words = pickle.load(f)
        f = open(os.path.join("léxico/",lexico_1+"_sent_words_polarity.p"), "rb")
        sent_words_polarity = pickle.load(f)
        
        f = open(os.path.join("léxico/",lexico_2+"_sent_words.p"), "rb")
        sent_words_lexico_2 = pickle.load(f)
        f = open(os.path.join("léxico/",lexico_2+"_sent_words_polarity.p"), "rb")
        sent_words_polarity_lexico_2 = pickle.load(f)

        f = open(os.path.join("léxico/",lexico_3+"_sent_words.p"), "rb")
        sent_words_lexico_3 = pickle.load(f)
        f = open(os.path.join("léxico/",lexico_3+"_sent_words_polarity.p"), "rb")
        sent_words_polarity_lexico_3 = pickle.load(f)

        #Busca palavra no segundo léxico
        for word in sent_words_lexico_2:
            if word not in sent_words:
                sent_words.append(word)
                #cria dicionário com polaridade
                sent_words_polarity[word] = sent_words_polarity_lexico_2[word]

        #busca palavra no terceiro léxico
        for word in sent_words_lexico_3:
            if word not in sent_words:
                sent_words.append(word)
                #cria dicionário com polaridade
                sent_words_polarity[word] = sent_words_polarity_lexico_3[word]
                
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)   

    l_sentimento= []
    word_sentimento = []
    w = []
    #busca polaridade de cada palavra do review
    for word in review:
        w = [word,int(sent_words_polarity.get(word,0))]
        word_sentimento.append(w)

    #retorna lista palavra e a polaridade de cada palavra
    return (word_sentimento)

def avaliacao(TP, TN, FP, FN, acertos):
    if TP == 0:
        print("********** ACURACIA **********\n")
        acuracia = ((TP+TN)/(TP+TN+FP+FN))*100
        print("\t\t",acuracia,"\t\t\n\n")
    else:
        print("********** COBERTURA **********\n")
        cobertura = TP / (TP+FN)
        print("\t\t",cobertura,"\t\t\n\n")

        print("********** PRECISÃO **********\n")
        precisao = TP / (TP+FP)
        print("\t\t",precisao,"\t\t\n\n")

        print("********** MÉDIA F **********\n")
        mediaf = 2 *((precisao * cobertura) / (precisao + cobertura))
        print("\t\t",mediaf,"\t\t\n\n")

        print("********** ACURACIA **********\n")
        acuracia = ((TP+TN)/(TP+TN+FP+FN))*100
        print("\t\t",acuracia,"\t\t\n\n")

    
            
def CBL(all_reviews):
    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    intensificacao = ['mais','muito','demais','completamente','absolutamente','totalmente','definitivamente','extremamente','frequentemente','bastante']
    reducao = ['pouco','quase','menos','apenas']
    
    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file:  #Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
    result_review = []
    
    #faz chamada da biblioteca spacy e atribui a uma variável
    spc = spacy.load('pt_core_news_sm')
    tratados = []

    #print(polarity_reviews,"\n\n")
    #print(all_reviews,"\n\n")
    sent_words = Sentilex()
    cont = 0

    
    #verifica cada review
    for review in all_reviews:

        review= str(review)
        #atribuindo o texto ao modelo spacy
        words = spc(review)


        #dando split no texto
        words.text.split()
        lista = []

        #retira pontuação e espaços do review
        for i,palavra in enumerate(words):
            if not palavra.is_punct:
                if not palavra.is_space:
                    plvra = palavra.text 
                    lista.append(plvra)

        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        #frase_polarity = lexico_sentimento_SentWordNetPT(lista)
                    
                    
        #REALIZAR AVALIAÇÃO COM SENTILEX
        #frase_polarity = lexico_sentimento_SentiLex(lista)

        #REALIZAR AVALIAÇÃO COM LIWC
        frase_polarity = lexico_sentimento_LIWC(lista)

        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(lista)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        #frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', lista)
        
        #print(frase_polarity)
        print("\n")


        
        janela = []
        sent_text = 0.0
        polaridade = 0.0
        #pega cada palavra de sentimento do review
        for i,termo in enumerate(frase_polarity):
                
                
            #busca termo com polaridade
            if(termo[1] != 0 ):
                x=i+3
                #janela.append(termo)
                posteriores = frase_polarity[i:x] #fatia a lista para pegar termo posterior
                #print(posteriores)
                #janela.append(posterior1)
                
                polaridade = float(termo[1])
                
                #print(posteriores)
                for j,item in enumerate(posteriores):
                    #print(item)
                    if item[0] in intensificacao:
                        if item[0] in negacao:
                            polaridade = polaridade/3
                            #print("polaridade 1:\t",polaridade)
                        else:
                            polaridade = polaridade*3
                            #print("polaridade 2:\t",polaridade)
                    if item[0] in reducao:
                        if item[0] in negacao:
                            polaridade = polaridade/3
                            #print("polaridade 3:\t",polaridade)
                        else:
                            polaridade = polaridade*3
                            #print("polaridade 4:\t",polaridade)
                    if item[0] in negacao and item[1] == 1:
                        polaridade = -1 * polaridade
                        #print("polaridade 5:\t",polaridade)

                sent_text = sent_text + polaridade

        print(sent_text)

        polaridade_rev = sent_text
        

        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1.0:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1.0:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0.0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)

        print("REVIEW Nº:\t",cont)
            
    acertos = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    zeros = 0
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i] and result_review[i] == 1.0:
            TP += 1

        if int(polarity[1]) == result_review[i] and result_review[i] == -1.0:
            TN += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == 1.0:
            FP += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == -1.0:
            FN += 1
   
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
            
        if int(polarity[1]) == 0 or  result_review[i] == 0.0:
            
            zeros += 1 #conta reviews que deram 0   
        else:
            print("")

    print("TP: ",TP,"\tTN: ",TN,"\tFP: ",FP,"\tFN: ",FN)
    print("\nTOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    #acuracia = acertos/(len(all_reviews))*100
    #print("\n\n\n\n\nacuracia:\t",acuracia,"%")
    avaliacao(TP, TN, FP, FN, acertos)
    

all_reviews = []


with open(os.path.join("Processed_Reviews.p"), "rb") as file: #->Processed_Reviews
        all_reviews = pickle.load(file)

CBL(all_reviews)


