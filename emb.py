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
            print("SIMILARES: \n")
            aspects_list.append(out[0][0].lower())
            print("Similar 1:",out[0][0])
            aspects_list.append(out[1][0].lower())
            print("Similar 2:",out[1][0])
            aspects_list.append(out[2][0].lower())
            print("Similar 3:",out[2][0],"\n")
    aspects_list = list(set(aspects_list))
    print("AQUI",aspects_list)
    if save:
        with open(os.path.join("Aspectos",seeds_type+"_embedding_aspects.p"), "wb") as f:
            pickle.dump(aspects_list, f)
            f.close()
    return aspects_list





file = open(os.path.join("Aspectos","ontology_explicit_aspects.p"), "rb")
asp_exp = pickle.load(file)
for line in asp_exp:
    for word in line:
        create_aspects_lexicon_embeddings(word, "ontol")


#printando model
file = open(os.path.join("word2v.model"), "rb")
m = pickle.load(file)
print("#############ASPECTOS SALVOS NO TREINAMENTO MEU########### \n",m)


#printando model
file = open(os.path.join("word2vec.model"), "rb")
model = pickle.load(file)
print("#############ASPECTOS SALVOS NO TREINAMENTO EMPRESTADO########### \n",model)


#printando os aspectos similares que salva dos embeddings
file = open(os.path.join("examples","noun_embedding_aspects.p"), "rb")
emb_exp = pickle.load(file)
print("#############ASPECTOS EMBEDDING ONTOLOGIA DE EXEMPLO########### \n",emb_exp)


#printando os aspectos similares que salva dos embeddings
file = open(os.path.join("Aspectos","ontol_embedding_aspects.p"), "rb")
emb_exp = pickle.load(file)
print("#############ASPECTOS EMBEDDING ONTOLOGIA QUE EU FIZ########### \n",emb_exp)


