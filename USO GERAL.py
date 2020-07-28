# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:54:31 2019

@author: yagoa
"""
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
from ftfy import fix_encoding
import subprocess
from collections import Counter

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



with open(os.path.join("léxico/","SentWordNet_PTBR200.p"), "rb") as f:
    sent_words = pickle.load(f)
with open(os.path.join("léxico/","SentWordNet_PTBR_polarity200.p"), "rb") as f:
    sent_words_polarity = pickle.load(f)

print("sent_words: \n",sent_words)
print("sent_words_polarity: \n",sent_words_polarity)
