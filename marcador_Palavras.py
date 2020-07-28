#coding: utf-8 
"""
Created on Thu Jan  2 10:00:06 2014

@author: Yagoa
"""

#from xml.etree import ElementTree as ET_xml
#from lxml import etree as ET_lxml

#BLIBLIOTECAS GERAIS
import pprint
import pickle
import string
import gensim
import fnmatch
import enelvo
import re
from enelvo import normaliser


#BIBLIOTECA PARA LER SENTIWORDNET-PT-BR
import pandas as pd

#BIBLIOTECAS DO NLTK
import nltk
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
import os
import io

#BIBLIOTECAS PARA UTILIZAR O SPACY
import spacy
from spacy import tokens

#BIBLIOTECAS PARA XML
import untangle
from xml.dom.minidom import parse
from bs4 import BeautifulSoup
nltk.download('rslp')
import xml.etree.ElementTree as ET
from ftfy import fix_encoding

#pre processa texto
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



VecIndSent =[]
ListAtribPal = [[]]
ListaPalSent =[[[]]]

for dirpath, _, files in os.walk("./Corpus Buscape/Analisados/teste"): #Pasta que se encontra o review analisado pelo Palavras
    for filename in files:
        if filename.endswith('.xml'):
                
            print(dirpath,filename)
            endereco = dirpath + "/" + filename
            
            """
            #Segunda forma de iterar sobre arquivo XML, utilizando o BeautifulSoup
            with open(endereco) as fp:
                soup = BeautifulSoup(fp, 'xml')
                
            print(soup.find_all('corpus'))
            """
            
            """
            #Terceira forma de iterar sobre arquivo XML, utilizando o Untangle
            endereco = dirpath + "/" + filename
            print(endereco)
            obj = untangle.parse(endereco)

            print(obj.corpus.body.s['id'])

            """
            
            parser = ET.XMLParser(encoding="utf-8")
            
            
            
            stemmer = nltk.stem.RSLPStemmer()
            
        
            print(endereco)
            
            #CÓDIGO SÓ ESTÁ RODANDO POR CONTA DO "iso-8859-5", SE VOLTAR iso-8859-5 PARA UTF8 MOSTRA O ERRO
            doc = ET.parse(endereco, parser=ET.XMLParser(encoding = 'iso-8859-5')) #parser=ET.XMLParser(encoding = 'utf-8-sig')
            
            
            root=doc.getroot()

            
            for body in root:
                
                for child in body:
                    x=child.attrib['id']
                    
                    VecIndSent.append(x)
                    node3 = child.find('graph')
                    
                    for node4 in node3:
                        
                        if node4.tag == 'terminals':
                            for node5 in node4:
                                
                                temp=[]
                                temp.append(node5.attrib["id"])
                                temp.append(pre_processing_text(node5.attrib["word"]))
                                temp.append(node5.attrib["lemma"])
                                temp.append(node5.attrib["pos"])
                                ListAtribPal.append(temp)
                                
                            ListaPalSent.append(ListAtribPal)
                            
            ListAtribPal = ListAtribPal[1:]
            print(ListAtribPal)
            ListAtribPal_dec =[]
            for a,word in enumerate(ListAtribPal):
                
                wrd1 = fix_encoding(word[1])
                wrd2 = fix_encoding(word[2])
                temp = [word[0],wrd1,wrd2,word[3]]
                ListAtribPal_dec.append(temp)

            print(ListAtribPal_dec)
            #print(ListAtribPal)
            
        
        
            
