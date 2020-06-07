# -*- coding: utf-8 -*-

import pickle
import os
import nltk
import gensim
import numpy as np
from enelvo import normaliser
import fnmatch




def create_aspects_lexicon_embeddings(seeds, seeds_type, number_synonym=3,save=False):
    #print("1")
    aspects_list = []
    model = gensim.models.Word2Vec.load("word2v.model") #open(os.path.join("Aspectos","ontology_explicit_aspects.p"), "rb")
    for word in seeds:
        aspects_list.append(word)
        if word in model.wv.vocab:
            print("WORD: ", word)
            out = model.wv.most_similar(positive=word, topn=number_synonym)
            print("\nSIMILARES: \n")
            aspects_list.append(out[0][0].lower())
            print("Similar 1:",out[0][0])
            aspects_list.append(out[1][0].lower())
            print("Similar 2:",out[1][0])
            aspects_list.append(out[2][0].lower())
            print("Similar 3:",out[2][0],"\n")
    aspects_list = list(set(aspects_list))
    print("CONTEÚDO DA LISTA: ",aspects_list)
    if save:
        with open(os.path.join("Aspectos",seeds_type+"_embedding_aspects.p"), "wb") as f:
            pickle.dump(aspects_list, f)
            f.close()
            
    return aspects_list



asp_exp = []
file = open(os.path.join("Aspectos","SUBS.p"), "rb")
asp_exp = pickle.load(file)
for word in asp_exp:
        create_aspects_lexicon_embeddings(word, "subs")

file.close()

#modelo de treinamento do exemplo
print("#############WORD2VEC.MODEL SALVOS NO TREINAMENTO EXEMPLO########## \n")
file = open(os.path.join("word2vec.model"), "rb")
model = pickle.load(file)
print(model)
file.close()

#aspectos base do exemplo
print("#############ASPECTOS DO EXEMPLO########### \n")
asp_e = []
file = open(os.path.join("examples","explicit_aspects.p"), "rb")
asp_e = pickle.load(file)
print(asp_e)
file.close()

asp_imp = []
file = open(os.path.join("examples","implicit_aspects.p"), "rb")
asp_imp = pickle.load(file)
print(asp_imp)
file.close()

#aspectos extraídos dos embeddings
print("#############ASPECTOS EMBEDDING EXTRAÍDO DE ONTOLOGIA DE EXEMPLO########### \n")
file = open(os.path.join("examples","ontology_embedding_aspects.p"), "rb")
emb_exp = pickle.load(file)
print(emb_exp)
file.close()

#modelo de treinamento que eu fiz
print("#############WORD2V.MODEL, ASPECTOS SALVOS NO MEU TREINAMENTO###########")
file = open(os.path.join("word2v.model"), "rb")
m = pickle.load(file)
print(m)
file.close()
      
print("#############ASPECTOS DE SUBSTANTIVOS########### \n")
asp_ex = []
file = open(os.path.join("aspectos","SUBS.p"), "rb")
asp_ex = pickle.load(file)
print(asp_ex)
file.close()
      
print("#############ASPECTOS EMBEDDING EXTRAÍDO DA ONTOLOGIA QUE EU FIZ########### \n")
emb_exp = []
#printando os aspectos similares que salva dos embeddings
file = open(os.path.join("Aspectos","subs_embedding_aspects.p"), "rb")
emb_exp = pickle.load(file)
print(emb_exp)
file.close()
      
