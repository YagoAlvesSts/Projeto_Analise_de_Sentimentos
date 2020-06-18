# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:54:31 2019

@author: yagoa
"""
from pathlib import Path
import pickle

#lendo arquivo com informações de polaridade
sentilexpt = open("SentiLex-PT02/SentiLex-flex-PT02.txt",'r',encoding="utf8")

#realiza tratamento de texto
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
try:    #verifica se léxico já foi lido uma vez e salvo
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
                sent_words_polarity[word] = polarity
    if save: #salva léxico
        with open(os.path.join("léxico/","SentiLex_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/","SentiLex_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)


print(dic_palavra_polaridade) #retorna dicionário com polaridades





def atribuir_sentimento(frase):
    frase = frase.lower()

    l_sentimento= []
    l = []
    word_sentimento = []
    w = []
    review_sentimento = []
    r = []

    for word in frase.split(): #verifica cada palavra de review
        #busca polaridade da palavra no dicionário de léxicos
        l_sentimento.append(int(dic_palavra_polaridade.get(word,0))) 

        #atribui palavra e polaridade a lista
        w = [word,int(dic_palavra_polaridade.get(word,0))]
        word_sentimento.append(w) #adiciona lista a lista com todas as palavras do review                     
        print(w)
        
    print(word_sentimento)
    score = sum(l_sentimento)
             
    r =[frase,score]
    review_sentimento.append(r)

    #with open("USO_GERAL1.p", "wb") as f:
        #pickle.dump(all_reviews, f)
    
    

    return (review_sentimento)



all_reviews=[]
with(open("USO_GERAL1.p", "rb")) as f:
    all_reviews = pickle.load(f)

with(open("USO_GERAL.p", "rb")) as f:
    all_rev = pickle.load(f)

print("ALL_REV",all_rev)

for review in all_reviews:
    #print("\n",review,"\n")
    atribuir_sentimento(review)
        
  


    
