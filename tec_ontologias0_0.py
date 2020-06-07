import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')



import pickle
import os
import nltk
import gensim
import numpy as np
from enelvo import normaliser
import fnmatch
import string



def create_aspects_lexicon_ontologies():
    """Create a list of the aspects indicated in the groups file"""
    explicit_aspects = []
    implicit_aspects = []
    with open("groups.xml", "r", encoding="utf8") as file:
        text = file.readlines()
        
        for line in text:
            if "Explicit" in line:
                word = line.split(">")[1].split("<")[0].split(",")
                text = [[words for words in sentences.lower().split(',')] for sentences in word]
                explicit_aspects.append(text)
            elif "Implicit" in line:
                word = line.split(">")[1].split("<")[0].split(",")
                text = [[words for words in sentences.lower().split(',')] for sentences in word]
                implicit_aspects.append(text)
    print(explicit_aspects)
    print(implicit_aspects)
    # Do some cleaning rules
    _explicit_aspects = []
    _implicit_aspects = []

    for aspect in explicit_aspects:
        if aspect != "s/n" and aspect != " . ":
            _explicit_aspects.append(aspect)
    for aspect in implicit_aspects:
        if aspect != "s/n" and aspect != " . ":
            _implicit_aspects.append(aspect)
            
    #_explicit_aspects = list(set(_explicit_aspects))
    #_implicit_aspects = list(set(_implicit_aspects))
    with open(os.path.join("Aspectos","ontology_explicit_aspects.p"), "wb") as f:
        pickle.dump(_explicit_aspects, f)
    with open(os.path.join("Aspectos","ontology_implicit_aspects.p"), "wb") as f:
        pickle.dump(_implicit_aspects, f)

    
    aex = []
    file = open(os.path.join("Aspectos","ontology_explicit_aspects.p"), "rb")
    asp_exp = pickle.load(file)

    """
    i=0
    j=0
    
    for i in range(len(asp_exp)):
        for j in range(i):
            aex.append(asp_exp[i][j])
            print(asp_exp[i][j])

    with open(os.path.join("Aspectos","ontology_explicit_aspects.p"), "wb") as f:
        pickle.dump(aex, f)
    """
    #print(asp_exp)

        
    aim = []
    file = open(os.path.join("Aspectos","ontology_implicit_aspects.p"), "rb")
    asp_imp = pickle.load(file)


    """
    i=0
    j=0
    
    for i in range(len(asp_imp)):
        for j in range(i):
            aex.append(asp_imp[i][j])

    with open(os.path.join("Aspectos","ontology_implicit_aspects.p"), "wb") as f:
        pickle.dump(aim, f)
     """   
    #print(asp_imp)
    
             
    return [_explicit_aspects, _implicit_aspects]



print(create_aspects_lexicon_ontologies())


