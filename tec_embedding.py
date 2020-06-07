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
    
    model = gensim.models.Word2Vec.load("word2v.model")
    """
    file = open(os.path.join("Aspectos","SUBSfreq.p"), "rb")
    seeds = pickle.load(file)
    """
    file = open(os.path.join("Aspectos","ontology_explicit_aspects.p"), "rb")
    s1 = pickle.load(file)
    file2 = open(os.path.join("Aspectos","ontology_implicit_aspects.p"), "rb")
    s2 = pickle.load(file2)

    print(s1)
    print(s2)
    seeds = s1+s2

    print(seeds)
    
    
    
    for word in seeds:
        
        aspects_list.append(word)
        if word in model.wv.vocab:
            out = model.wv.most_similar(positive=word, topn=number_synonym)
            aspects_list.append(out[0][0].lower())
            aspects_list.append(out[1][0].lower())
            aspects_list.append(out[2][0].lower())
    aspects_list = list(set(aspects_list))
    if save:
        with open(os.path.join("Aspectos",seeds_type+"_embedding_aspects.p"), "wb") as f:
            pickle.dump(aspects_list, f)
            arquivo = open("Aspectos/"+seeds_type+"_embendding_aspects.txt","w")
            arquivo.write(' '.join(aspects_list))
            arquivo.close()
            
    print(aspects_list)
    return aspects_list

create_aspects_lexicon_embeddings('ontol')
