# -*- coding: utf-8 -*-

import pickle
import os
import nltk
import gensim
import numpy as np
from enelvo import normaliser
import fnmatch




def create_aspects_lexicon_embeddings( seeds_type, number_synonym=3,save=True):
    aspects_list = []
    s1 = []
    s2 = []
    
    model = gensim.models.Word2Vec.load("word2v.model")#faz leitura do modelo de treinamento criado anteriormente
    """
    file = open(os.path.join("Aspectos","SUBSfreq.p"), "rb")
    seeds = pickle.load(file)
    """
    #lê aspectos extraídos de ontologias
    file = open(os.path.join("Aspectos","noun_aspects_TreeTagger.p"), "rb")
    s1 = pickle.load(file)
    file2 = open(os.path.join("Aspectos","noun_aspects_TreeTagger.p"), "rb")
    s2 = pickle.load(file2)

    print(s1)
    print(s2)
    seeds = s1+s2 #concatena implícitos e explícitos

    print(seeds)
    
    
    #busca cada palavra do review
    for word in seeds:
        
        aspects_list.append(word)#atribui palavra a lista de aspectos
        if word in model.wv.vocab: #busca palavra em vocabulário de treinamento
            #busca 3 palavras mais próximas de palavra semente
            out = model.wv.most_similar(positive=word, topn=number_synonym)
            aspects_list.append(out[0][0].lower())#1º palavra mais próxima
            aspects_list.append(out[1][0].lower())#2º palavra mais próxima
            aspects_list.append(out[2][0].lower())#3º palavra mais próxima
    aspects_list = list(set(aspects_list))#adiciona palavras a lista de aspectos
    if save:#salva lista de aspectos em .txt e .p
        with open(os.path.join("Aspectos",seeds_type+"_embedding_aspects_treetagger.p"), "wb") as f:
            pickle.dump(aspects_list, f)
            arquivo = open("Aspectos/"+seeds_type+"_embendding_aspects_treetagger.txt","w")
            arquivo.write(' '.join(aspects_list))
            arquivo.close()
            
    print(aspects_list)
    return aspects_list#retorna lista de aspectos

create_aspects_lexicon_embeddings('ontol')
