# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:54:31 2019

@author: yagoa
"""
from pathlib import Path
import pickle

#lendo arquivo com informações de polaridade
sentilexpt = open("SentiLex-PT01/SentiLex-lem-PT01.txt",'r',encoding="utf8")


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
for i in sentilexpt.readlines():
    pos_ponto = i.find('.')
    palavra = pre_processing_text((i[:pos_ponto]))
    pol_pos = i.find('POL')
    polaridade = (i[pol_pos+4:pol_pos+6]).replace(';','')
    dic_palavra_polaridade[palavra] = polaridade


print(dic_palavra_polaridade)





def atribuir_sentimento(frase):
    frase = frase.lower()

    l_sentimento= []
    l = []
    word_sentimento = []
    w = []
    review_sentimento = []
    r = []

    for word in frase.split():
        l_sentimento.append(int(dic_palavra_polaridade.get(word,0)))

        
        w = [word,int(dic_palavra_polaridade.get(word,0))]
        word_sentimento.append(w)                      
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
        
  


    
