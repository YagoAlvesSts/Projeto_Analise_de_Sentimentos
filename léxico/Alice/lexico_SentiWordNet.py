import nltk
import os
import pickle
import pandas as pd
import spacy
from spacy import tokens
import pprint
import string
import gensim
import fnmatch
import enelvo
import re
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

"""
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
            print(termo[3],trm,word)
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
        

all_reviews = []

with open(os.path.join("USO_GERAL.p"), "rb") as file:
        all_reviews = pickle.load(file)


tec_posicao_adjetivo_spacy(all_reviews)


  




