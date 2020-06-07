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


def create_aspects_lexicon_ontologies(save=True):
    """Create a list of the aspects indicated in the groups file"""
    explicit_aspects = []
    implicit_aspects = []
    with open("groups.xml", "r", encoding="utf8") as file:
        text = file.readlines()
        for line in text:
            if "Explicit" in line:
                for word in line.split(">")[1].split("<")[0].split(","):
                    explicit_aspects.append(pre_processing_text(word))
            elif "Implicit" in line:
                for word in line.split(">")[1].split("<")[0].split(","):
                    implicit_aspects.append(pre_processing_text(word))
    # Do some cleaning rules
    _explicit_aspects = []
    _implicit_aspects = []
    for aspect in explicit_aspects:
        if aspect != "s/n" and aspect != " . ":
            _explicit_aspects.append(aspect)
    for aspect in implicit_aspects:
        if aspect != "s/n" and aspect != " . ":
            _implicit_aspects.append(aspect)
    # Remove repetition on aspects list
    _explicit_aspects = list(set(_explicit_aspects))
    _implicit_aspects = list(set(_implicit_aspects))
    if save:
        with open(os.path.join("Aspectos","ontology_explicit_aspects.p"), "wb") as f:
            pickle.dump(_explicit_aspects, f)
            arquivo = open('Aspectos/ontology_explicit_aspects.txt','w')
            arquivo.write(' '.join(_explicit_aspects))
            arquivo.close()
        with open(os.path.join("Aspectos","ontology_implicit_aspects.p"), "wb") as f:
            pickle.dump(_implicit_aspects, f)
            arquivo = open('Aspectos/ontology_implicit_aspects.txt.txt','w')
            arquivo.write(' '.join(_implicit_aspects))
            arquivo.close()

    """
    aex = []
    file = open(os.path.join("Aspectos","ontology_explicit_aspects.p"), "rb")
    asp_exp = pickle.load(file)

    
    i=0
    j=0
    
    for i in range(len(asp_exp)):
        aex.append(asp_exp[i])

    with open(os.path.join("Aspectos","ontology_explicit_aspects.p"), "wb") as f:
        pickle.dump(aex, f)
    
    aim = []
    file = open(os.path.join("Aspectos","ontology_implicit_aspects.p"), "rb")
    asp_imp = pickle.load(file)

    i=0
    j=0
    
    for i in range(len(asp_imp)):
            aim.append(asp_imp[i])

    with open(os.path.join("Aspectos","ontology_implicit_aspects.p"), "wb") as f:
        pickle.dump(aim, f)
   
    print(aim)
    print("####################################")
    print(_implicit_aspects)
    """
    
    return [_explicit_aspects, _implicit_aspects]



print(create_aspects_lexicon_ontologies())


