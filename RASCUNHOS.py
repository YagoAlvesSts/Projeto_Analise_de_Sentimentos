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


    input_chars = ["\n"," / ", " - ", "|","@","#", "$", "%", "&", "*", "(", ")", "[", "]", "{", "}", ";", ":", "<", ">", "=", "_", "+",]
    output_chars = [".","/", "-", "","","","","","","","","","","","","","","","","","","","",]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text


all_reviews = []
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/testando/positivo"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review = f.read()
        review = pre_processing_text(review, use_normalizer=False)
        all_reviews.append(review)
with open("rt_polarity_pos.p", "wb") as f:
    pickle.dump(all_reviews, f)

arquivo = open('rt_polarity_pos.txt','w')
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

    input_chars = ["\n", ".", "!", "?", " / ", " - ", "|"]
    output_chars = [" . ", " . ", " . ", " . ",  "/", "-", ""]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text


all_reviews_n = []
all_reviews_p = []
all_reviews_t = []

for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/negativos"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review_n = f.read()
        review_n = [pre_processing_text(review_n, use_normalizer=True)]
        all_reviews_n.append(review_n)

for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/positivos"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review_p = f.read()
        review_p = [pre_processing_text(review_p, use_normalizer=True)]
        all_reviews_p.append(review_p)


all_reviews_t = all_reviews_n + all_reviews_p
print(all_reviews_t)

with open("Processed_Reviews.p", "wb") as f:
    pickle.dump(all_reviews_t, f)

"""

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
for dirpath, _, files in os.walk("./Corpus Buscape/testando"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review = f.read()
        review = pre_processing_text(review)

        #rev = removerCaracteresEspeciais(review)

        #rev = re.sub('[#$%^&*()[]{};:,<>\`~=_+]', ' ', review)
        
        all_reviews.append(review)
        
arquivo = open('reviews_for_P.txt','w')
arquivo.write('\n'.join(all_reviews))
arquivo.close()


# -*- coding: utf-8 -*-

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

def lexico_sentimento_SentiLex(review):
  

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
            try:
                polarity = line[1].split('N0=')[1].split(';')[0]
            except:
                polarity = line[1].split('N1=')[1].split(';')[0]
            sent_words.append(word)
            dic_palavra_polaridade[word] = polarity

    l_sentimento= []
    word_sentimento = []
    w = []

    for word in review:
        w = [word,int(dic_palavra_polaridade.get(word,0))]
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

        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        #frase_polarity = lexico_sentimento_SentWordNetPT(lista)
                    
                    
        #REALIZAR AVALIAÇÃO COM SENTILEX
        frase_polarity = lexico_sentimento_SentiLex(lista)
        #print(frase_polarity)
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

        #print(wd)
        #print("\n")
        
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


with open(os.path.join("Processed_Reviews.p"), "rb") as file: #->Processed_Reviews
        all_reviews = pickle.load(file)

tec_posicao_adjetivo_nltk(all_reviews) #whit SentWordNet






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


    input_chars = [" / ", " - ", "|","@","#", "$", "%", "&", "*", "(", ")", "[", "]", "{", "}", ";", ":", "<", ">", "=", "_", "+",]
    output_chars = ["/", "-", "","","","","","","","","","","","","","","","","","","","",]

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
cont = 0
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/testando/negativo"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review = f.read()
        review = pre_processing_text(review)
        
        #rev = removerCaracteresEspeciais(review)

        #rev = re.sub('[#$%^&*()[]{};:,<>\`~=_+]', ' ', review)
        
        arquivo = open("Corpus Buscape/for_palavras/"+str(cont)+".txt","w",encoding="utf8")
        arquivo.write(''.join(review)+'\ n')
        arquivo.close()
        cont = cont+1




# -*- coding: utf-8 -*-

import pprint
import pickle
import nltk
import string
import os
import gensim
import fnmatch
import enelvo
import re

import pandas as pd

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

    sentilexpt = open("SentiLex-PT02/SentiLex-flex-PT02.txt",'r',encoding="utf8")
    dic_palavra_polaridade = {}
    for i in sentilexpt.readlines():
        pos_ponto = i.find('.')
        palavra = pre_processing_text((i[:pos_ponto]))
        pol_pos = i.find('POL')
        polaridade = (i[pol_pos+4:pol_pos+6]).replace(';','')
        dic_palavra_polaridade[palavra] = polaridade

    #retorna dicionário do SentiLex
    return(dic_palavra_polaridade)


def lexico_sentimento_SentWordNetPT(review):
   

    word_sentimento = []
    
    pol = 0
    word_pol = []
    #print(SentiWordNet)
    for word in review:
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
        if termo[3] == word:
            scorepos = scorepos + float(termo[1])
            scoreneg = scoreneg + float(termo[2])
            
            print("termo:",termo[3],"\tposscore: ",scorepos,"\tnegscore: ",scoreneg)

            return (scorepos,scoreneg)
        
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
        com_sentilex = lexico_sentimento_SentiLex(lista)
        print("polaridades com SentiLex:\n",com_sentilex)

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

with open(os.path.join("USO_GERAL.p"), "rb") as file:
        all_reviews = pickle.load(file)


tec_posicao_adjetivo_spacy(all_reviews)



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

    l_sentimento= []
    word_sentimento = []
    w = []

    for word in review:
        w = [word,int(dic_palavra_polaridade.get(word,0))]
        word_sentimento.append(w)

    #retorna lista palavra e a polaridade de cada palavra
    return (word_sentimento)

def Sentilex():

    sentilexpt = open("SentiLex-PT02/SentiLex-flex-PT02.txt",'r',encoding="utf8")
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

for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/lexico"):
    for filename in fnmatch.filter(files, '*.txt'):
            f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
            review = f.read()
            print(review)
            review = pre_processing_text(review, use_normalizer=True)
            all_reviews.append(review)
    with open("tec_linha_de_base.p", "wb") as f:
        pickle.dump(all_reviews, f) Processed_Reviews 

with open(os.path.join("USO_GERAL.p"), "rb") as file:
        all_reviews = pickle.load(file)


tec_posicao_adjetivo_spacy(all_reviews)




def tec_posicao_adjetivo_nltk(all_reviews):

    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    

    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file: #-.Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
        
    result_review = []

    
    
    #CHAMA DICIONARIO DO SENTILEX Processed_Reviews_polarity
    #print("DICIONARIO SENTILEX: \n",Sentilex())
    sent_words = Sentilex()
    cont = 0
    
    #pega um review de cada vez
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
        #frase_polarity = lexico_sentimento_SentWordNetPT(words)
        
        #REALIZAR AVALIAÇÃO COM SENTILEX
        #frase_polarity = lexico_sentimento_SentiLex(words)
        
        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(words)
        
        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(words)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', words)
        
        
        
        #print("\nPALAVRAS COM POLARIDADE:\n",frase_polarity)

        #realizar marcação de termos da lista
        words = nltk.pos_tag(words)

        
        #print("\n",words)

        
        #pega cada palavra do review
        for i,termo in enumerate(words):
            apont = i
            anterior=words[apont-1]
            #busca termos que são substantivos
            if(termo[1] == 'NN' or termo[1] == 'NNS'):
                x=i+2
                posterior = words[i+1:x] #fatia a lista para pegar termo posterior
                #print("Verificou substantivo\n")
                
                for wrd,ps in posterior:
                    post = wrd
                
                if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                    palavra = termo[0]
                    #busca palavra na lista que contém polaridade
                    for j,item in enumerate(frase_polarity):
                        
                        #print("Entrou negação")
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        
                        for wd,p in poeio:
                            pt = wd
                        #se palavra de negação existe, atribui polaridade negativa   
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                elif(anterior[1]=='JJ'): #verifica se termo anterior é adjetivo
                    #print(anterior[0])
                    #print("Entrou Adjetivo antes\n")
                    palavra = termo[0]
                    ant = anterior[0]
                    #print(ant)

                    #busca palavra na lista que contém polaridade
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wd,p in poeio:
                            pt = wd
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("Pegou")
                        #atribui polaridade do adjetivo ao substantivo    
                        if palavra == item[0]:
                            #print("Atribuiu",polaridade_adj)
                            item[1] = polaridade_adj
                            
                            
                       
                
                for wrd,ps in posterior:
                    
                    #verifica se termo posterior é adjetivo
                    if(ps == 'JJ'):
                        polaridade_adj_pos=0
                        #print("Entrou adjetivo depois")
                        #print(wrd)
                                                
                        palavra = termo[0]
                        ant = anterior[0]

                        #busca palavra na lista que contém polaridade 
                        for k,item in enumerate(frase_polarity):
                            ap = k
                            
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                            
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                
                                
                            #pega polaridade do adjetivo
                            if wrd == item[0]:
                                polaridade_adj_pos = item[1]
                                #print("Pegou")
                                #if frase_polarity[k-1]

                            #atribui polaridade do adjetivo ao substantivo      
                            for g,item in enumerate(frase_polarity):
                                if palavra == item[0] and ant == antes:
                                    #print("Atribuiu depois",polaridade_adj_pos)
                                    item[1] = polaridade_adj_pos
                        
                           
                
        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
        polaridade_rev = 0
        #soma polaridades do review após aplicar técnica
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
        
        print("\n",polaridade_rev)

        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)

        print("REVIEW Nº:\t",cont)
            
    acertos = 0
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
        else:
            print("")
    
                
    print("\nTOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    acuracia = acertos/(len(all_reviews))*100
    print("\n\n\n\n\nacuracia:\t",acuracia,"%")



# -*- coding: utf-8 -*-

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


def lexico_sentimento_SentWordNetPT(review):
   

    word_sentimento = []
    
    pol = 0
    word_pol = []
    #print(SentiWordNet)
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
        if(polaridade == None):#se palavra não existe no léxico, atribui 0
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
    #busca palavra no léxico
    for i,termo in enumerate(SentiWordNet):
        trm = pre_processing_text(termo[3]) #trata a palavra a ser buscada
        if trm == word: #compara palavra do review com o léxico
            #aqui o ideal seria somar, mas atualmente ele só pega polaridade da ultima palavra encontrada
            scorepos = scorepos + float(termo[1]) #soma polaridades positivas
            scoreneg = scoreneg + float(termo[2]) #soma polaridades negativas

            
            #print("termo:",termo[3],"\tposscore: ",scorepos,"\tnegscore: ",scoreneg)

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


def tec_posicao_adjetivo_nltk(all_reviews):

    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    

    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file: #-.Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
        
    result_review = []

    
    
    #CHAMA DICIONARIO DO SENTILEX Processed_Reviews_polarity
    #print("DICIONARIO SENTILEX: \n",Sentilex())
    sent_words = Sentilex()
    cont = 0
    
    #pega um review de cada vez
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
        #frase_polarity = lexico_sentimento_SentWordNetPT(words)
        
        #REALIZAR AVALIAÇÃO COM SENTILEX
        frase_polarity = lexico_sentimento_SentiLex(words)
        
        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(words)
        
        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(words)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        #frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', words)
        
        
        
        #print("\nPALAVRAS COM POLARIDADE:\n",frase_polarity)

        #realizar marcação de termos da lista
        words = nltk.pos_tag(words)

        
        #print("\n",words)

        
        #pega cada palavra do review
        for i,termo in enumerate(words):
            apont = i
            anterior=words[apont-1]
            #busca termos que são substantivos
            if(termo[1] == 'NN' or termo[1] == 'NNS'):
                x=i+2
                posterior = words[i+1:x] #fatia a lista para pegar termo posterior
                #print("Verificou substantivo\n")
                
                for wrd,ps in posterior:
                    post = wrd
                
                if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                    palavra = termo[0]
                    #busca palavra na lista que contém polaridade
                    for j,item in enumerate(frase_polarity):
                        
                        #print("Entrou negação")
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        
                        for wd,p in poeio:
                            pt = wd
                        #se palavra de negação existe, atribui polaridade negativa   
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                elif(anterior[1]=='JJ'): #verifica se termo anterior é adjetivo
                    #print(anterior[0])
                    #print("Entrou Adjetivo antes\n")
                    palavra = termo[0]
                    ant = anterior[0]
                    #print(ant)

                    #busca palavra na lista que contém polaridade
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wd,p in poeio:
                            pt = wd
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("Pegou")
                        #atribui polaridade do adjetivo ao substantivo    
                        if palavra == item[0]:
                            #print("Atribuiu",polaridade_adj)
                            item[1] = polaridade_adj
                            
                            
                       
                
                for wrd,ps in posterior:
                    
                    #verifica se termo posterior é adjetivo
                    if(ps == 'JJ'):
                        polaridade_adj_pos=0
                        #print("Entrou adjetivo depois")
                        #print(wrd)
                                                
                        palavra = termo[0]
                        ant = anterior[0]

                        #busca palavra na lista que contém polaridade 
                        for k,item in enumerate(frase_polarity):
                            ap = k
                            
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                            
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                
                                
                            #pega polaridade do adjetivo
                            if wrd == item[0]:
                                polaridade_adj_pos = item[1]
                                #print("Pegou")
                                #if frase_polarity[k-1]

                            #atribui polaridade do adjetivo ao substantivo      
                            for g,item in enumerate(frase_polarity):
                                if palavra == item[0] and ant == antes:
                                    #print("Atribuiu depois",polaridade_adj_pos)
                                    item[1] = polaridade_adj_pos
                        
                           
                
        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
        polaridade_rev = 0
        #soma polaridades do review após aplicar técnica
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
        
        print("\n",polaridade_rev)

        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)

        print("REVIEW Nº:\t",cont)
            
    acertos = 0
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
        else:
            print("")
    
                
    print("\nTOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    acuracia = acertos/(len(all_reviews))*100
    print("\n\n\n\n\nacuracia:\t",acuracia,"%")

            
def tec_posicao_adjetivo_spacy(all_reviews):
    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    
    
    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file:  #Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
    result_review = []
    
    #faz chamada da biblioteca spacy e atribui a uma variável
    spc = spacy.load('pt_core_news_sm')
    tratados = []

    
    
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
        frase_polarity = lexico_sentimento_SentiLex(lista)

        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(lista)

        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(lista)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        #frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', lista)
        
       # print(frase_polarity)
        print("\n")

        
        tagger = []
        wd = []

        #realiza marcação de cada palavra
        for i,word in enumerate(words):
            #print(i,word)
            
            if not word.is_punct:
                if not word.is_space:
                    #print(word.text, word.pos_)
                    tagger = [word.text, word.pos_] #pega palavra e pos tagger
                    wd.append(tagger) #atribui palavra e pos tagger a lista
            if i > len(words):
                break

        #print(wd)
        print("\n")

        #pega cada palavra do review
        for j,termo in enumerate(wd):
            apont = j
                
            anterior=wd[j-1]
                
            #print("ANTERIOR: ",anterior)
            #print("PALAVRA \t",j,termo)
            
            #busca termos que são substantivos
            if(termo[1] == 'NOUN'):
                #print("\n****** ENTROU \t 1 ******\n", termo)
                x=j+2
                    
                posterior = wd[j+1:x] #fatia a lista para pegar termo posterior
                #print("POSTERIOR", posterior)
                #print("Verificou substantivo\n")
                        
                for wrd,ps in posterior:
                    post = wrd
                        
                #print("verificando se: ",anterior[0]," é negação")

                #busca se termo anterior é palavra de negação    
                if anterior[0] in negacao:
                    #print("\n****** ENTROU \t 2 ******\n", anterior)
                    #print("verificou negação")
                    palavra = termo[0]
                    #busca palavra na lista que contém polaridade
                    for l,item in enumerate(frase_polarity):
                        ap = l
                        y = l+2
                                
                        poeio=frase_polarity[l+1:y]
                                
                        for w,p in poeio:
                            pt = w
                        #print(palavra,item[0])
                        #se palavra de negação existe, atribui polaridade negativa
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                    
                    
                if(anterior[1]=='ADJ'): #verifica se termo anterior é adjetivo
                    ant = anterior[0]
                    #print("\n****** ENTROU \t 3 ******\n", anterior)
                    #print("Entrou Adjetivo antes\n")
                    #print(ant)
                    #print(anterior[1])
                    palavra = termo[0]
                        
                        
                    #busca palavra na lista que contém polaridade    
                    for k,item in enumerate(frase_polarity):
                        #print("\n****** ENTROU \t 4 ******\n", item)
                        ap = k
                            
                        z = k+2
                            
                        poeio=frase_polarity[k+1:z]
                            
                        for w,p in poeio:
                            pt = w

                        #pega polaridade do adjetivo        
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("\n****** ENTROU \t 5 ******\n", item)

                        #atribui polaridade do adjetivo ao substantivo          
                        if palavra == item[0]:
                            item[1] = polaridade_adj
                            #print("\n****** ENTROU \t 6 ******\n", item)
                                          
            
                for wrd,ps in posterior:
                    #verifica se termo posterior é adjetivo
                    if(ps == 'ADJ'):
                        #print("Entrou adjetivo depois",)                        
                        palavra = termo[0]
                        ant = anterior[0]

                        #busca palavra na lista que contém polaridade    
                        for k,item in enumerate(frase_polarity):
                            ap = k
                                
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                                
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                    
                                    
                            #print("VERIFICANDO DENTRO DA LISTA COM POLARIDADE")
                                #pega polaridade do adjetivo
                                if wrd == item[0]:
                                    polaridade_adj_pos = item[1]
                                    #print("entrou e recolheu polaridade1")
                                #atribui polaridade do adjetivo ao substantivo
                                for g,item in enumerate(frase_polarity):    
                                    if palavra == item[0]:
                                        item[1] = polaridade_adj_pos
                                        #print("entrou e atribuiu polaridade")
                        
                           
              
        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
        polaridade_rev = 0
        #soma polaridades do review após aplicar técnica
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
        
        print("\n",polaridade_rev,"\n")
        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1




            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)

        print("REVIEW Nº:\t",cont)
            
    acertos = 0
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
        else:
            print("")
    
                
    print("TOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    acuracia = acertos/(len(all_reviews))*100
    print("\n\n\n\n\nacuracia:\t",acuracia,"%")
    

all_reviews = []

for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/lexico"):
    for filename in fnmatch.filter(files, '*.txt'):
            f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
            review = f.read()
            print(review)
            review = pre_processing_text(review, use_normalizer=True)
            all_reviews.append(review)
    with open("tec_linha_de_base.p", "wb") as f:
        pickle.dump(all_reviews, f) Processed_Reviews 


with open(os.path.join("Processed_Reviews.p"), "rb") as file: #->Processed_Reviews
        all_reviews = pickle.load(file)

tec_posicao_adjetivo_spacy(all_reviews)


import os
import pickle

texto = "Alice é linda.\nAlice é legal.\nAlice é feia tbm."
file =  open(os.path.join("notepad.txt"), "w", encoding="utf8" )
file.writelines(texto)    
file.close()  


WINDOWS_LINE_ENDING = b'\r\n'
UNIX_LINE_ENDING = b'\n'

# relative or absolute file path, e.g.:
file_path = r"C:\Users\Alice\Desktop\UFG\Projeto\Aprendendo\Projeto_1\notepad.txt"

with open(file_path, 'rb') as open_file:
    content = open_file.read()

content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

with open(file_path, 'wb') as open_file:
    open_file.write(content)





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


"""
