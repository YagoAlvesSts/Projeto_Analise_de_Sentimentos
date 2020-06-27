# -*- coding: utf-8 -*-

import gensim
import numpy as np
import os
import pickle
import fnmatch
from enelvo import normaliser
from unidecode import unidecode
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.corpus import stopwords
import re
from unicodedata import normalize
import codecs

def pre_processing_text(text, use_normalizer=False):

    if use_normalizer:
        norm = normaliser.Normaliser()
        text = norm.normalise(text)


    input_chars = [" \n","..."," / ","/", " - ","-", "|","@","#", "$", "%", "&", "*", "(", ")", "[", "]", "{", "}", ";", ":", "<", ">", "=", "_", "+",]
    output_chars = [". \n",".","/", " ","-"," ", "","","","","","","","","","","","","","","","","","","","",]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text


def adicionarLF():
    
    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'
    #C:\Users\yagoa\Desktop\Faculdade\PP\Projeto_Analise_de_Sentimentos\Corpus Buscape\for_palavras
    folder = os.listdir(r"C:/Users/yagoa/Desktop/Faculdade/PP/Projeto_Analise_de_Sentimentos/Corpus Buscape/for_palavras")

    for file in folder:

        with open("Corpus Buscape/for_palavras/"+file, 'rb') as open_file:
            content = open_file.read()

        content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

        with open("Corpus Buscape/for_palavras/"+file, 'wb') as open_file:
            open_file.write(content)



WINDOWS_LINE_ENDING = b'\r\n'
UNIX_LINE_ENDING = b'\n'
cont = 0    #C:\Users\yagoa\Desktop\Faculdade\PP\Projeto_Analise_de_Sentimentos\Corpus Buscape\testando\negativo

#print("foi")
for dirpath, _, files in os.walk("./Corpus Buscape/testando/negativo"):
    #print("FOI")
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review = f.read()
        #print(review)
        review = pre_processing_text(review)
        
        #adiciona ponto no final da sentença
        for i,carac in enumerate(review):
            
            anterior = review[i-1]
                
            #print(apont)    
            if (carac == '\n') :
                x=i+2
                posterior = review[i+1:x]
                #print(posterior)
                #print("final da linha")
                if (anterior != '.'):
                    review.replace('\n','. \n')
                    break
                    #print("É PONTO")
       
            
        #rev = re.sub('[#$%^&*()[]{};:,<>\`~=_+]', ' ', review)
        
        #print(review)
        arquivo = open("Corpus Buscape/for_palavras/id_"+str(cont)+".txt","w",encoding="utf8")
        arquivo.write(''.join(review))
        arquivo.close()
        cont = cont+1



adicionarLF()
    




