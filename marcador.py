"""
# -*- coding: utf-8 -*-

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


with(open("tagger/text_for_tagger.p", "rb")) as f:
    all_reviews = pickle.load(f)
#print(all_reviews)

# Criando e escrevendo em arquivos de texto (modo 'w').
arquivo = open('matriz01.txt','w')
arquivo.writelines(all_reviews)
arquivo.close()


# Lendo o arquivo criado:
arquivo = open('matriz01.txt','r')
conteudo=arquivo.read()
    

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

with open(os.path.join("tagger","tagger_nltk.p"), "wb") as f:
        pickle.dump(list(words), f)
        
with(open("tagger/tagger_nltk.p", "rb")) as f:
    all_reviews = pickle.load(f)
print(all_reviews)
"""
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

def lexico_sentimento_SentiLex(review,save=True):
  
    try:
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
                try:
                    polarity = line[1].split('N0=')[1].split(';')[0]
                except:
                    polarity = line[1].split('N1=')[1].split(';')[0]
                sent_words.append(word)
                sent_words_polarity[word] = polarity
    if save:
        with open(os.path.join("léxico/","SentiLex_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/","SentiLex_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)
            
    l_sentimento= []
    word_sentimento = []
    w = []

    for word in review:
        w = [word,int(sent_words_polarity.get(word,0))]
        word_sentimento.append(w)

    #retorna lista palavra e a polaridade de cada palavra
    return (word_sentimento)

def Sentilex():

    sentilexpt = open("SentiLex-PT02/SentiLex-flex-PT02.txt",'r',encoding="utf8")
    dic_palavra_polaridade = {}
    text = sentilexpt.readlines()
    sent_words = []
    for line in text:
            line = line.split(',')
            word = line[0]
            word = pre_processing_text(word)
            #word = N.unidecode(word) #tira acentuação
            try:
                polarity = line[1].split('N0=')[1].split(';')[0]
            except:
                polarity = line[1].split('N1=')[1].split(';')[0]
            sent_words.append(word)
            dic_palavra_polaridade[word] = polarity

    #retorna dicionário do SentiLex
    return(dic_palavra_polaridade)


def lexico_sentimento_SentWordNetPT(review):
   

    word_sentimento = []
    
    pol = 0
    word_pol = []
    #print(SentiWordNet)
    for word in review:
        word = pre_processing_text(word)
        polaridade = atribui_polaridade_sentiwordnet(word)
        if(polaridade !=  None ):
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
            word_sentimento.append(word_pol)
        if(polaridade == None):
            word_pol = [word,'0']
            word_sentimento.append(word_pol)
        
    return (word_sentimento)


def atribui_polaridade_sentiwordnet(word):
    SentiWordNet = []
    df = pd.read_csv('léxico/SentiWordNet_PT/SentiWord Pt-BR v1.0b.txt', delimiter="\t", header=None, names=["ID","PosScore", "NegScore", "Termo"])
    #print(df.values)
    SentiWordNet = df.values
    scorepos = 0.0
    scoreneg = 0.0
    for i,termo in enumerate(SentiWordNet):
        trm = pre_processing_text(termo[3])
        if trm == word:
            scorepos = scorepos + float(termo[1])
            scoreneg = scoreneg + float(termo[2])
            
            #print("termo:",termo[3],"\tposscore: ",scorepos,"\tnegscore: ",scoreneg)

            return (scorepos,scoreneg)

def lexico_sentimento_LIWC(review, save=True):
    
    try:
        with open(os.path.join("léxico/","LIWC_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/","LIWC_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        sent_words = []
        sent_words_polarity = {}
        with open("léxico/liwc.txt", encoding="utf8") as f:
            text = f.readlines()
            for line in text:
                word = line.split()[0]
                #print("Line",line)
                
                word = pre_processing_text(word)
                #word = line.split()[0]
                if "126" in line:
                    # Positive sentiment word
                    sent_words.append(word)
                    sent_words_polarity[word] = "1"
                if "127" in line:
                    sent_words.append(word)
                    sent_words_polarity[word] = "-1"
                else:
                    sent_words.append(word)
                    sent_words_polarity[word] = "0"
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
    for termo in review:
        w = [termo,int(sent_words_polarity.get(termo,0))]
        word_sentimento.append(w)

    return (word_sentimento)

def lexico_sentimento_OpLexicon(review):
    try:
        with open(os.path.join("léxico/","OpLexicon_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/","OpLexicon_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        print("\nProcessing Lexico")
        sent_words = []
        sent_words_polarity = {}
        f = open("léxico/lexico_v3.0.txt",'r',encoding="utf8")
        text = f.readlines()
        for line in text:
            line = line.split(',')
            word = line[0]
            word = pre_processing_text(word)
            #word = N.unidecode(word) #tira acentuação
            polarity = line[2]
            sent_words.append(word)
            sent_words_polarity[word] = polarity
        
    with open(os.path.join("léxico/","OpLexicon_sent_words.p"), "wb") as f:
        pickle.dump(sent_words, f)
    with open(os.path.join("léxico/","OpLexicon_sent_words_polarity.p"), "wb") as f:
        pickle.dump(sent_words_polarity, f)

    l_sentimento= []
    word_sentimento = []
    w = []
    for termo in review:
        w = [termo,int(sent_words_polarity.get(termo,0))]
        word_sentimento.append(w)

    return (word_sentimento)

def concatenar(lexico_1, lexico_2, lexico_3, review, save=True):

    try:
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        lexico_sentimento_OpLexicon(review)
        lexico_sentimento_SentiLex(review)
        lexico_sentimento_LIWC(review)

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

        for word in sent_words_lexico_2:
            if word not in sent_words:
                sent_words.append(word)
                sent_words_polarity[word] = sent_words_polarity_lexico_2[word]

        for word in sent_words_lexico_3:
            if word not in sent_words:
                sent_words.append(word)
                sent_words_polarity[word] = sent_words_polarity_lexico_3[word]
                
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)   

    l_sentimento= []
    word_sentimento = []
    w = []
    for word in review:
        w = [word,int(sent_words_polarity.get(word,0))]
        word_sentimento.append(w)

    #retorna lista palavra e a polaridade de cada palavra
    return (word_sentimento)


def tec_posicao_adjetivo_nltk(all_reviews):

    all_tokenized_reviews = []
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    

    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file: #-.Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
        
    result_review = []

    
    
    #CHAMA DICIONARIO DO SENTILEX Processed_Reviews_polarity
    #print("DICIONARIO SENTILEX: \n",Sentilex())
    sent_words = Sentilex()
    cont = 0
    
    
    for i,review in enumerate(all_reviews):

        review = ''.join(review)
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


        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        frase_polarity = lexico_sentimento_SentWordNetPT(words)
        
        #REALIZAR AVALIAÇÃO COM SENTILEX
        #frase_polarity = lexico_sentimento_SentiLex(words)
        
        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(words)
        
        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(words)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        #frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', words)
        
        
        
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
                        
                        #print("Entrou negação")
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        
                        for wd,p in poeio:
                            pt = wd
                           
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                elif(anterior[1]=='JJ'):
                    #print(anterior[0])
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
                            polaridade_adj=p

                            
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("Pegou")
                            
                        if palavra == item[0]:
                            #print("Atribuiu",polaridade_adj)
                            item[1] = polaridade_adj
                            
                            
                       
                
                for wrd,ps in posterior:
                    
                    
                    if(ps == 'JJ'):
                        polaridade_adj_pos=0
                        #print("Entrou adjetivo depois")
                        #print(wrd)
                                                
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
                                #print("Pegou")
                                #if frase_polarity[k-1]
                                
                            for g,item in enumerate(frase_polarity):
                                if palavra == item[0] and ant == antes:
                                    #print("Atribuiu depois",polaridade_adj_pos)
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

        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        #frase_polarity = lexico_sentimento_SentWordNetPT(lista)
                    
                    
        #REALIZAR AVALIAÇÃO COM SENTILEX
        #frase_polarity = lexico_sentimento_SentiLex(lista)

        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(lista)

        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(lista)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', lista)
        
        print(frase_polarity)
        print("\n")

        
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

        print(wd)
        print("\n")
        
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
                        #print(palavra,item[0])           
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
                                for g,item in enumerate(frase_polarity):    
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

with open(os.path.join("Processed_Reviews.p"), "rb") as file: #->Processed_Reviews
        all_reviews = pickle.load(file)

tec_posicao_adjetivo_nltk(all_reviews)



