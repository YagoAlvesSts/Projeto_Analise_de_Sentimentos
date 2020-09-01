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

#atribui polaridade com Sentilex
def lexico_sentimento_SentiLex(review,save=True):
  
    try:#verifica se léxico já foi lido uma vez e salvo
        with open(os.path.join("léxico/","SentiLex_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/","SentiLex_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        #lendo arquivo com informações de polaridade
        sentilexpt = open("SentiLex-PT02/SentiLex-flex-PT02.txt",'r',encoding="utf8")
        sent_words_polarity = {}

        text = sentilexpt.readlines()
        sent_words = []
        for line in text:
                line = line.split(',')
                word = line[0]
                word = pre_processing_text(word)
                #word = N.unidecode(word) #tira acentuação
                try:#busca polaridade
                    polarity = line[1].split('N0=')[1].split(';')[0]
                except:
                    polarity = line[1].split('N1=')[1].split(';')[0]
                sent_words.append(word)
                #cria dicionário com polaridades
                sent_words_polarity[word] = polarity
    if save:#salva léxico
        with open(os.path.join("léxico/","SentiLex_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/","SentiLex_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)
            
    l_sentimento= []
    word_sentimento = []
    w = []
    #busca polaridade de cada palavra do review
    for word in review:
        w = [word,int(sent_words_polarity.get(word,0))]
        word_sentimento.append(w)#adiciona palavra com polaridade em lista

    #retorna lista palavra e a polaridade de cada palavra
    return (word_sentimento)

def Sentilex():
    #lendo arquivo com informações de polaridade
    sentilexpt = open("SentiLex-PT02/SentiLex-flex-PT02.txt",'r',encoding="utf8")
    dic_palavra_polaridade = {}
    text = sentilexpt.readlines()
    sent_words = []
    for line in text:
            line = line.split(',')
            word = line[0]
            word = pre_processing_text(word)
            #word = N.unidecode(word) #tira acentuação
            try:#busca polaridade
                polarity = line[1].split('N0=')[1].split(';')[0]
            except:
                polarity = line[1].split('N1=')[1].split(';')[0]
            sent_words.append(word)
            #cria dicionário com polaridades
            dic_palavra_polaridade[word] = polarity

    #retorna dicionário do SentiLex
    return(dic_palavra_polaridade)


def lexico_sentimento_SentWordNetPT(review,save=True):


    try: #verifica se léxico já foi lido uma vez e salvo
        with open(os.path.join("léxico/","SentWordNet_PTBR.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/","SentWordNet_PTBR_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        
        sent_words_polarity = {}
        sent_words = []
        
        pol = 0
        word_pol = []
        #print(SentiWordNet)
        #atribui polaridade a cada palavra do review
        for word in review:
            word = pre_processing_text(word) #trata o texto
            #chama função de atribuir polaridade a palavra
            polaridade = atribui_polaridade_sentiwordnet(word)
            
            if(polaridade !=  None ):#se polaridade existir no léxico, atribui ao score
                #print()
                scorepos = polaridade[0]
                scoreneg = polaridade[1]
                if(scorepos > scoreneg):
                    pol = '1'
                elif(scorepos < scoreneg):
                    pol = '-1'
                else:
                    pol = '0'
                
                word_pol = [word,pol]
                #cria dicionário com palavra e polaridade
                sent_words_polarity[word] = pol
                
                #adiciona palavra e polaridade a lista
                sent_words.append(word_pol)
                
            if(polaridade == None):#se palavra não existe no léxico, atribui 0
                word_pol = [word,'0']
                #adiciona palavra e polaridade a lista
                sent_words.append(word_pol)
                
                #cria dicionário com palavra e polaridade
                sent_words_polarity[word] = '0'

            
    if save:#salva léxico
        with open(os.path.join("léxico/","SentWordNet_PTBR.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/","SentWordNet_PTBR_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)
            
    word_sentimento = []
    #busca cada palavra do review
    for word in review:
        wrd = [word,int(sent_words_polarity.get(word,0))] #pega palavra com polaridade no dicionario
        word_sentimento.append(wrd)
        
    return (word_sentimento) #retorna review com polaridades


def atribui_polaridade_sentiwordnet(word):
    
    sent_words_polarity = {}
    #lista que será adicionado valores do léxico
    SentiWordNet = []
    sent_words =[]
    #lê léxico com o pandas
    df = pd.read_csv('léxico/SentiWordNet_PT/SentiWord Pt-BR v1.0b.txt', delimiter="\t", header=None, names=["ID","PosScore", "NegScore", "Termo"])
    #print(df.values)
    SentiWordNet = df.values #pega valores do léxico de polaridades
    scorepos = 0.0
    scoreneg = 0.0
    neg = []
    pos = []
    cont = 0
    #busca palavra no léxico
    for i,termo in enumerate(SentiWordNet):
        trm = pre_processing_text(termo[3]) #trata a palavra a ser buscada
        if trm == word: #compara palavra do review com o léxico
            cont += 1
            #print(termo[3],trm,word)
            #aqui o ideal seria somar, mas atualmente ele só pega polaridade da ultima palavra encontrada
                
            neg.append(float(termo[1]))
            pos.append(float(termo[2]))


                
            #scorepos = scorepos + float(termo[1])#soma polaridades positivas
            #scoreneg = scoreneg + float(termo[2])#soma polaridades negativas
                
            #print("\tposscore: ",pos,"\tnegscore: ",neg)
            
                
        scoreneg = sum(i for i in neg)

        scorepos = sum(j for j in pos)
        
            
    #print(scorepos,scoreneg)    
    return (scorepos,scoreneg) #retorna score de polaridades

def lexico_sentimento_LIWC(review, save=True):
    
    try: #verifica se léxico já foi lido uma vez e salvo
        with open(os.path.join("léxico/","LIWC_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/","LIWC_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        sent_words = []
        sent_words_polarity = {}
        #busca arquivo com léxico LIWC
        with open("léxico/liwc.txt", encoding="utf8") as f:
            text = f.readlines()
            for line in text: 
                word = line.split()[0]
                #print("Line",line)
                
                word = pre_processing_text(word)
                #word = line.split()[0]
                #BUSCA POLARIDADE
                if "126" in line:
                    # Positive sentiment word
                    sent_words.append(word)
                    sent_words_polarity[word] = "1" #cria dicionário de léxico com polaridade
                if "127" in line:
                    sent_words.append(word)
                    sent_words_polarity[word] = "-1" #cria dicionário de léxico com polaridade
                else:
                    sent_words.append(word)
                    sent_words_polarity[word] = "0" #cria dicionário de léxico com polaridade
        # Remove duplicated words
        sent_words = list(set(sent_words))
    if save:
        with open(os.path.join("léxico/","LIWC_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/","LIWC_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)

    l_sentimento= []
    word_sentimento = []
    w = []

    #busca polaridade de cada palavra do review
    for termo in review: 
        w = [termo,int(sent_words_polarity.get(termo,0))]
        word_sentimento.append(w)

    return (word_sentimento) #retorna lista com palavras e polaridades

#Atribui polaridade com léxico OpLexicon
def lexico_sentimento_OpLexicon(review):
    try: #verifica se léxico já foi lido uma vez e salvo
        with open(os.path.join("léxico/","OpLexicon_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/","OpLexicon_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        print("\nProcessing Lexico")
        sent_words = []
        sent_words_polarity = {}
        #busca arquivo com léxico OpLexicon
        f = open("léxico/lexico_v3.0.txt",'r',encoding="utf8")
        text = f.readlines()
        #BUSCA POLARIDADE
        for line in text:
            line = line.split(',')
            word = line[0]
            word = pre_processing_text(word)
            #word = N.unidecode(word) #tira acentuação
            polarity = line[2]
            sent_words.append(word)
            sent_words_polarity[word] = polarity #cria dicionário com polaridade
        
    with open(os.path.join("léxico/","OpLexicon_sent_words.p"), "wb") as f:
        pickle.dump(sent_words, f)
    with open(os.path.join("léxico/","OpLexicon_sent_words_polarity.p"), "wb") as f:
        pickle.dump(sent_words_polarity, f)

    l_sentimento= []
    word_sentimento = []
    w = []
    #busca polaridade de cada palavra do review
    for termo in review:
        w = [termo,int(sent_words_polarity.get(termo,0))]
        word_sentimento.append(w)

    return (word_sentimento) #retorna lista com palavras e polaridades

#função que concatena léxicos de polaridade
def concatenar(lexico_1, lexico_2, lexico_3, review, save=True):

    try: #busca se léxicos já foram concatenados anteriormente
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:#chama cada léxico separadamente para depois concatenar
        lexico_sentimento_OpLexicon(review)
        lexico_sentimento_SentiLex(review)
        lexico_sentimento_LIWC(review)

        #SALVA LÉXICOS QUE FORAM ABERTOS ATRAVÉS DE CHAMADA DE FUNÇÃO
        f = open(os.path.join("léxico/",lexico_1+"_sent_words.p"), "rb")
        sent_words = pickle.load(f)
        f = open(os.path.join("léxico/",lexico_1+"_sent_words_polarity.p"), "rb")
        sent_words_polarity = pickle.load(f)
        
        f = open(os.path.join("léxico/",lexico_2+"_sent_words.p"), "rb")
        sent_words_lexico_2 = pickle.load(f)
        f = open(os.path.join("léxico/",lexico_2+"_sent_words_polarity.p"), "rb")
        sent_words_polarity_lexico_2 = pickle.load(f)

        f = open(os.path.join("léxico/",lexico_3+"_sent_words.p"), "rb")
        sent_words_lexico_3 = pickle.load(f)
        f = open(os.path.join("léxico/",lexico_3+"_sent_words_polarity.p"), "rb")
        sent_words_polarity_lexico_3 = pickle.load(f)

        #Busca palavra no segundo léxico
        for word in sent_words_lexico_2:
            if word not in sent_words:
                sent_words.append(word)
                #cria dicionário com polaridade
                sent_words_polarity[word] = sent_words_polarity_lexico_2[word]

        #busca palavra no terceiro léxico
        for word in sent_words_lexico_3:
            if word not in sent_words:
                sent_words.append(word)
                #cria dicionário com polaridade
                sent_words_polarity[word] = sent_words_polarity_lexico_3[word]
                
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("léxico/",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)   

    l_sentimento= []
    word_sentimento = []
    w = []
    #busca polaridade de cada palavra do review
    for word in review:
        w = [word,int(sent_words_polarity.get(word,0))]
        word_sentimento.append(w)

    #retorna lista palavra e a polaridade de cada palavra
    return (word_sentimento)



def TreeTagger(texto): #Função para passar o texto e vai retornar a palavra o tag e o lemma
    
    file =  open(os.path.join("C:\TreeTagger", "texto.txt"), "w", encoding="utf8" )
    file.writelines(texto)    
    file.close()

    process = subprocess.Popen([r'\TreeTagger\executar.bat'],
                         shell = True,
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         universal_newlines=True)
    stdout, stderr = process.communicate()
    #print(stdout)
    result = stdout.split('\n')
    pos_tag = []
    for x in result:
        word = fix_encoding(x)
        pos_tag.append(word.split('\t'))
        
    return pos_tag


def most_commom(lst):
    data = Counter(lst)
    return(data.most_common())

def pre_processamento(text):

    str(text)
    text = text.lower()

    input_chars = ["\n", ".", "!", "?", " / ", " - ", "|", '``', "''"]
    output_chars = [" . ", " . ", " . ", " . ", "/", "-", "", "", ""]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text

def palavras(save="False"):
    count = 0
    analisados = []
    
    while count <= 1682:
        '''tive que pegar os arquivos desse jeito pq tem muitos arquivos dai na hora que ele abre ele pula do 0 para o 10000 mas pode abrir do jeito mais chic'''
        endereco = "Corpus Buscape/Analisados/Analisados/id_" + str(count) + "_Palavras.xml" 
        print(endereco)
        pos_tags = []
        
        with open(endereco, "r", errors='ignore') as file: #não usei encoding utf8 pq alguns da erro corrigi os caracteres com o fix_encoding
            text = file.readlines()
            for line in text:
                if "<t id=" in line:
                    pos_tag = []
                    aux = line.split('<t id=')[1]
                    ident = line.split('<t id="')[1].split('" word=')[0]
                    word = line.split('word="')[1].split('" lemma=')[0]
                    word = fix_encoding(word)
                    lemma = line.split('lemma="')[1].split('" pos=')[0]
                    lemma = fix_encoding(lemma)
                    pos = line.split('pos="')[1].split('" morph=')[0]

                    pos_tag.append(ident)
                    pos_tag.append(word)
                    pos_tag.append(lemma)
                    pos_tag.append(pos)
                    
                    pos_tags.append(pos_tag)
        print(pos_tags)

        analisados.append(pos_tags)
        count += 1                  
        
    print(analisados)
    if save:
        with open(os.path.join("Corpus Buscape/Analisados/salvando_teste","analisados_processados.p"), "wb") as f:
            pickle.dump(analisados, f)

    return analisados



cobertura = 0
precisao = 0
mediaf = 0
acuracia = 0
def avaliacao(TP, TN, FP, FN, acertos):
    if TP == 0:
        print("********** ACURACIA **********\n")
        acuracia = ((TP+TN)/(TP+TN+FP+FN))*100
        print("\t\t",acuracia,"\t\t\n\n")
    else:
        print("********** COBERTURA **********\n")
        cobertura = TP / (TP+FN)
        print("\t\t",cobertura,"\t\t\n\n")

        print("********** PRECISÃO **********\n")
        precisao = TP / (TP+FP)
        print("\t\t",precisao,"\t\t\n\n")

        print("********** MÉDIA F **********\n")
        mediaf = 2 *((precisao * cobertura) / (precisao + cobertura))
        print("\t\t",mediaf,"\t\t\n\n")

        print("********** ACURACIA **********\n")
        acuracia = ((TP+TN)/(TP+TN+FP+FN))*100
        print("\t\t",acuracia,"\t\t\n\n")


def tec_aspecto_palavra_NLTK(all_reviews):

    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    

    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file: #-.Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
        
    result_review = []

    
    
    #CHAMA DICIONARIO DO SENTILEX Processed_Reviews_polarity
    #print("DICIONARIO SENTILEX: \n",Sentilex())
    sent_words = Sentilex()
    cont = 0
    
    #pega um review de cada vez
    for i,review in enumerate(all_reviews):

        review = ''.join(review)
        norm = normaliser.Normaliser()
        
        #normaliza a sentença
        norm_sentence = norm.normalise(review)
        
        #coloca toda sentença em minúsculo
        norm_sentence = norm_sentence.lower()
        
        #possibilita mudar o atributo que desejar
        norm.capitalize_inis = True

        temp = unidecode(norm_sentence)

        #divide as palavras em uma lista(split)
        tokens = word_tokenize(temp)

        #remove pontuação de cada palavra
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        #remove qualquer outro caracter que não seja do alfabeto
        words = [word for word in stripped if word.isalpha()]


        stop_words = set(stopwords.words('portuguese'))
        words = [w for w in words if not w in stop_words]


        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        #frase_polarity = lexico_sentimento_SentWordNetPT(words)
        
        #REALIZAR AVALIAÇÃO COM SENTILEX
        frase_polarity = lexico_sentimento_SentiLex(words)
        
        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(words)
        
        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(words)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        #frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', words)
        
        
        
        #print("\nPALAVRAS COM POLARIDADE:\n",frase_polarity)

        #realizar marcação de termos da lista
        words = nltk.pos_tag(words)

        
        #print("\n",words)
        frase_polarity2 = frase_polarity

        existe = False
        #pega cada palavra do review
        for i,termo in enumerate(words):
            apont = i
            anterior=words[apont-1]
            
            #busca termos que são substantivos
            
            if(termo[1] == 'NN' or termo[1] == 'NNS'):
                x=i+2
                posterior = words[i+1:x] #fatia a lista para pegar termo posterior
                #print("Verificou substantivo\n")
                post = ''
                for wrd,ps in posterior:
                    post = wrd
                
                if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                    palavra = termo[0]
                    #busca palavra na lista que contém polaridade
                    for j,item in enumerate(frase_polarity):
                        
                        #print("Entrou negação")
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        
                        for wd,p in poeio:
                            pt = wd
                        #se palavra de negação existe, atribui polaridade negativa   
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                
                if post in negacao: #busca se termo anterior é palavra de negação
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wd,p in poeio:
                                pt = wd
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0]:
                                item[1] = '-1'

                                
                elif(anterior[1]=='JJ'): #verifica se termo anterior é adjetivo
                    #print(anterior[0])
                    #print("Entrou Adjetivo antes\n")
                    existe = True
                    palavra = termo[0]
                    ant = anterior[0]
                    #print(ant)

                    #busca palavra na lista que contém polaridade
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wd,p in poeio:
                            pt = wd
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("Pegou")
                        #atribui polaridade do adjetivo ao substantivo    
                        if palavra == item[0]:
                            #print("Atribuiu",polaridade_adj)
                            item[1] = polaridade_adj
                            
                            
                       
                
                for wrd,ps in posterior:
                    
                    #verifica se termo posterior é adjetivo
                    if(ps == 'JJ'):
                        existe = True
                        polaridade_adj_pos=0
                        #print("Entrou adjetivo depois")
                        #print(wrd)
                                                
                        palavra = termo[0]
                        ant = anterior[0]

                        #busca palavra na lista que contém polaridade 
                        for k,item in enumerate(frase_polarity):
                            ap = k
                            
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                            
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                
                                
                            #pega polaridade do adjetivo
                            if wrd == item[0]:
                                polaridade_adj_pos = item[1]
                                #print("Pegou")
                                #if frase_polarity[k-1]

                            #atribui polaridade do adjetivo ao substantivo      
                            for g,item in enumerate(frase_polarity):
                                if palavra == item[0] and ant == antes:
                                    #print("Atribuiu depois",polaridade_adj_pos)
                                    item[1] = polaridade_adj_pos
                        
        if existe is False: #se não foi avaliado pela técnica
            #print("não existe adjetivo depois de substantivo!")
            frase_polarity = frase_polarity2 #realiza apenas soma das polaridades das palavras
            #print("trocou frase com polaridades:\n", frase_polarity2)
            
                
        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
        polaridade_rev = 0
        #soma polaridades do review após aplicar técnica
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
        
        print("\n",polaridade_rev)

        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)

        print("REVIEW Nº:\t",cont)
            
    acertos = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    zeros = 0
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i] and result_review[i] == 1.0:
            TP += 1

        if int(polarity[1]) == result_review[i] and result_review[i] == -1.0:
            TN += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == 1.0:
            FP += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == -1.0:
            FN += 1
   
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
            
        if int(polarity[1]) == 0 or  result_review[i] == 0.0:
            
            zeros += 1 #conta reviews que deram 0   
        else:
            print("")
    
    print("TP: ",TP,"\tTN: ",TN,"\tFP: ",FP,"\tFN: ",FN)            
    print("TOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    #acuracia = acertos/(len(all_reviews))*100
    #print("\n\n\n\n\nacuracia:\t",acuracia,"%")
    avaliacao(TP, TN, FP, FN, acertos)

"""
def tec_aspecto_adjetivo_NLTK(all_reviews):

    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    

    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file: #-.Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
        
    result_review = []

    
    
    #CHAMA DICIONARIO DO SENTILEX Processed_Reviews_polarity
    #print("DICIONARIO SENTILEX: \n",Sentilex())
    sent_words = Sentilex()
    cont = 0
    
    #pega um review de cada vez
    for i,review in enumerate(all_reviews):

        review = ''.join(review)
        norm = normaliser.Normaliser()
        
        #normaliza a sentença
        norm_sentence = norm.normalise(review)
        
        #coloca toda sentença em minúsculo
        norm_sentence = norm_sentence.lower()
        
        #possibilita mudar o atributo que desejar
        norm.capitalize_inis = True

        temp = unidecode(norm_sentence)

        #divide as palavras em uma lista(split)
        tokens = word_tokenize(temp)

        #remove pontuação de cada palavra
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        #remove qualquer outro caracter que não seja do alfabeto
        words = [word for word in stripped if word.isalpha()]


        stop_words = set(stopwords.words('portuguese'))
        words = [w for w in words if not w in stop_words]


        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        frase_polarity = lexico_sentimento_SentWordNetPT(words)
        
        #REALIZAR AVALIAÇÃO COM SENTILEX
        #frase_polarity = lexico_sentimento_SentiLex(words)
        
        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(words)
        
        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(words)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        #frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', words)
        
        
        
        #print("\nPALAVRAS COM POLARIDADE:\n",frase_polarity)

        #realizar marcação de termos da lista
        words = nltk.pos_tag(words)

        
        #print("\n",words)
        frase_polarity2 = frase_polarity

        existe = False
        #pega cada palavra do review
        for i,termo in enumerate(words):
            apont = i
            anterior=words[apont-1]
            
            #busca termos que são substantivos
            
            if(termo[1] == 'NN' or termo[1] == 'NNS'):
                x=i+2
                posterior = words[i+1:x] #fatia a lista para pegar termo posterior
                #print("Verificou substantivo\n")
                post = ''
                
                for wrd,ps in posterior:
                    post = wrd
                #print(post)
                if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                    palavra = termo[0]
                    #busca palavra na lista que contém polaridade
                    for j,item in enumerate(frase_polarity):
                        
                        #print("Entrou negação")
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        pt = ''
                        for wd,p in poeio:
                            pt = wd
                        #se palavra de negação existe, atribui polaridade negativa   
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                #print(post)
                #busca se termo posterior é palavra de negação
                if post in negacao: 
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wd,p in poeio:
                                pt = wd
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0]:
                                item[1] = '-1'       
                            
                elif(anterior[1]=='JJ'): #verifica se termo anterior é adjetivo
                    #print(anterior[0])
                    #print("Entrou Adjetivo antes\n")
                    existe = True
                    palavra = termo[0]
                    ant = anterior[0]
                    #print(ant)

                    #busca palavra na lista que contém polaridade
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wd,p in poeio:
                            pt = wd
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("Pegou")
                        #atribui polaridade do adjetivo ao substantivo    
                        if palavra == item[0]:
                            #print("Atribuiu",polaridade_adj)
                            item[1] = polaridade_adj
                            
                            
                       
                
                for wrd,ps in posterior:
                    
                    #verifica se termo posterior é adjetivo
                    if(ps == 'JJ'):
                        existe = True
                        polaridade_adj_pos=0
                        #print("Entrou adjetivo depois")
                        #print(wrd)
                                                
                        palavra = termo[0]
                        ant = anterior[0]

                        #busca palavra na lista que contém polaridade 
                        for k,item in enumerate(frase_polarity):
                            ap = k
                            
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                            
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                
                                
                            #pega polaridade do adjetivo
                            if wrd == item[0]:
                                polaridade_adj_pos = item[1]
                                #print("Pegou")
                                #if frase_polarity[k-1]

                            #atribui polaridade do adjetivo ao substantivo      
                            for g,item in enumerate(frase_polarity):
                                if palavra == item[0] and ant == antes:
                                    #print("Atribuiu depois",polaridade_adj_pos)
                                    item[1] = polaridade_adj_pos


        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
        polaridade_rev = 0
        adj_tem_polaridade = False
        pol_adj = []
        #print(existe)
        if existe is False: #se não foi avaliado por aspecto buscará polaridade dos adjetivos
            frase_polarity = frase_polarity2
            #print("verificou que não tem substantivo e entrou no primeiro False")
            for i,termo in enumerate(words):
                
                apont = i
                anterior=words[apont-1]
                
                #busca termos que são adjetivos
                
                if(termo[1] == 'JJ'):
                    x=i+2
                    posterior = words[i+1:x] #fatia a lista para pegar termo posterior
                    #print("Verificou adjetivo\n")
                    if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wd,p in poeio:
                                pt = wd
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0] and post == pt:
                                item[1] = '-1'
                            
                    if post in negacao: #busca se termo anterior é palavra de negação
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wd,p in poeio:
                                pt = wd
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0]:
                                item[1] = '-1'
                                
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wd,p in poeio:
                            pt = wd
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if termo[0] == item[0]:
                            polaridade_adj = item[1]
                            if polaridade_adj != '0':
                                adj_tem_polaridade = True
                                pol_adj.append(polaridade_adj)
                    
        
                    
        
        
            if adj_tem_polaridade is True:
                #print("tem adjetivo com polaridade!\n")
                for i in pol_adj:
                    polaridade_rev = int(i) + int(polaridade_rev)
                
                #print("\n",polaridade_rev)

                #busca se resultado da soma torna review positivo
                if polaridade_rev >= 1:
                    polaridade_rev = 1
                    cont +=1
                    result_review.append(1)

                #busca se resultado da soma torna review negativo    
                if polaridade_rev <= -1:
                    polaridade_rev = -1
                    cont +=1
                    result_review.append(-1)

                #busca se resultado da soma torna review neutro    
                if polaridade_rev == 0:
                    polaridade_rev = 0
                    cont +=1
                    result_review.append(0)

                print("REVIEW Nº:\t",cont)
                
            elif adj_tem_polaridade is False: #se não foi avaliado pela técnica
                #print("não tem adjetivo com polaridade!")
                frase_polarity = frase_polarity2 #realiza apenas soma das polaridades das palavras
                #print("trocou frase com polaridades:\n", frase_polarity2)
                polaridade_rev = 0
                #soma polaridades do review após aplicar técnica
                for item,pol in frase_polarity:
                    valor = int(pol)
                    polaridade_rev = polaridade_rev + valor
                
                #print("\n",polaridade_rev)

                #busca se resultado da soma torna review positivo
                if polaridade_rev >= 1:
                    polaridade_rev = 1
                    cont +=1
                    result_review.append(1)

                #busca se resultado da soma torna review negativo    
                if polaridade_rev <= -1:
                    polaridade_rev = -1
                    cont +=1
                    result_review.append(-1)

                #busca se resultado da soma torna review neutro    
                if polaridade_rev == 0:
                    polaridade_rev = 0
                    cont +=1
                    result_review.append(0)

    if existe is True:
        polaridade_rev = 0
        #soma polaridades do review após aplicar técnica
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
                
        print("\n",polaridade_rev)

        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)        
            
    acertos = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i] and result_review[i] == 1.0:
            TP += 1

        if int(polarity[1]) == result_review[i] and result_review[i] == -1.0:
            TN += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == 1.0 and int(polarity[1]) == -1.0:
            FP += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == -1.0 and int(polarity[1]) == 1.0:
            FN += 1
   
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
        else:
            print("")
    
    print("TP: ",TP,"\tTN: ",TN,"\tFP: ",FP,"\tFN: ",FN)            
    print("TOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    #acuracia = acertos/(len(all_reviews))*100
    #print("\n\n\n\n\nacuracia:\t",acuracia,"%")
    avaliacao(TP, TN, FP, FN, acertos)
"""    
            
def tec_aspecto_palavra_spacy(all_reviews):
    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    
    
    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file:  #Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
    result_review = []
    
    #faz chamada da biblioteca spacy e atribui a uma variável
    spc = spacy.load('pt_core_news_sm')
    tratados = []

    
    
    sent_words = Sentilex()
    cont = 0

    
    #verifica cada review
    for review in all_reviews:

        review= str(review)
        #atribuindo o texto ao modelo spacy
        words = spc(review)


        #dando split no texto
        words.text.split()
        lista = []

        #retira pontuação e espaços do review
        for i,palavra in enumerate(words):
            if not palavra.is_punct:
                if not palavra.is_space:
                    plvra = palavra.text 
                    lista.append(plvra)

        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        #frase_polarity = lexico_sentimento_SentWordNetPT(lista)
                    
                    
        #REALIZAR AVALIAÇÃO COM SENTILEX
        frase_polarity = lexico_sentimento_SentiLex(lista)

        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(lista)

        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(lista)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        #frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', lista)
        
        #print(frase_polarity)
        print("\n")
        frase_polarity2 = frase_polarity
        
        tagger = []
        wd = []

        #realiza marcação de cada palavra
        for i,word in enumerate(words):
            #print(i,word)
            
            if not word.is_punct:
                if not word.is_space:
                    #print(word.text, word.pos_)
                    tagger = [word.text, word.pos_] #pega palavra e pos tagger
                    wd.append(tagger) #atribui palavra e pos tagger a lista
            if i > len(words):
                break

        #print(wd)
        print("\n")
        existe = False
        #pega cada palavra do review
        for i,termo in enumerate(wd):
            apont = i
            #print(i)
            
            if i < len(wd):
                anterior=wd[apont-1]
                
            #print(anterior)
            #busca termos que são substantivos
            
            if(termo[1] == 'NOUN' ):
                x=i+2
                posterior = wd[i+1:x] #fatia a lista para pegar termo posterior
                #print("Verificou substantivo\n")
                post = ''
                
                
                #print(posterior)
                
                for token in posterior:
                    post = token[0]
                
                    
                #print(post)
                
                if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                    palavra = termo[0]
                    #busca palavra na lista que contém polaridade
                    for j,item in enumerate(frase_polarity):
                        
                        #print("Entrou negação")
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        
                        for wordss,p in poeio:
                            pt = wordss
                            
                        #se palavra de negação existe, atribui polaridade negativa   
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                
                if post in negacao: #busca se termo anterior é palavra de negação
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wordss,p in poeio:
                                pt = wordss
                                
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0]:
                                item[1] = '-1'

                #print(anterior)                
                if(anterior[1]=='ADJ'): #verifica se termo anterior é adjetivo
                    #print(anterior[0])
                    #print("Entrou Adjetivo antes\n")
                    existe = True
                    palavra = termo[0]
                    ant = anterior[0]
                    #print(ant)

                    #busca palavra na lista que contém polaridade
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wordss,p in poeio:
                            pt = wordss
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("Pegou")
                        #atribui polaridade do adjetivo ao substantivo    
                        if palavra == item[0]:
                            #print("Atribuiu",polaridade_adj)
                            item[1] = polaridade_adj
                            
                            
                       
                
                for wrd,ps in posterior:
                    
                    #verifica se termo posterior é adjetivo
                    if(ps == 'ADJ'):
                        existe = True
                        polaridade_adj_pos=0
                        #print("Entrou adjetivo depois")
                        #print(wrd)
                                                
                        palavra = termo[0]
                        ant = anterior[0]

                        #busca palavra na lista que contém polaridade 
                        for k,item in enumerate(frase_polarity):
                            ap = k
                            
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                            
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                
                                
                            #pega polaridade do adjetivo
                            if wrd == item[0]:
                                polaridade_adj_pos = item[1]
                                #print("Pegou")
                                #if frase_polarity[k-1]

                            #atribui polaridade do adjetivo ao substantivo      
                            for g,item in enumerate(frase_polarity):
                                if palavra == item[0] and ant == antes:
                                    #print("Atribuiu depois",polaridade_adj_pos)
                                    item[1] = polaridade_adj_pos
                        
        if existe is False: #se não foi avaliado pela técnica
            #print("não existe adjetivo depois de substantivo!")
            frase_polarity = frase_polarity2 #realiza apenas soma das polaridades das palavras
            #print("trocou frase com polaridades:\n", frase_polarity2)                           
              
        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
        polaridade_rev = 0
        #soma polaridades do review após aplicar técnica
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
        
        print("\n",polaridade_rev,"\n")
        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)

        print("REVIEW Nº:\t",cont)
            
    acertos = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    zeros = 0
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i] and result_review[i] == 1.0:
            TP += 1

        if int(polarity[1]) == result_review[i] and result_review[i] == -1.0:
            TN += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == 1.0:
            FP += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == -1.0:
            FN += 1
   
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
            
        if int(polarity[1]) == 0 or  result_review[i] == 0.0:
            
            zeros += 1 #conta reviews que deram 0   
        else:
            print("")
    
    print("TP: ",TP,"\tTN: ",TN,"\tFP: ",FP,"\tFN: ",FN)            
    print("TOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    #acuracia = acertos/(len(all_reviews))*100
    #print("\n\n\n\n\nacuracia:\t",acuracia,"%")
    avaliacao(TP, TN, FP, FN, acertos)


def tec_aspecto_palavra_TreeTagger(all_reviews):

    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    

    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file: #-.Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
        
    result_review = []

    
    
    #CHAMA DICIONARIO DO SENTILEX Processed_Reviews_polarity
    #print("DICIONARIO SENTILEX: \n",Sentilex())
    sent_words = Sentilex()
    cont = 0
    
    #pega um review de cada vez
    for i,review in enumerate(all_reviews):

        pos_tag = TreeTagger(review) #passo um review por parametro para a função tretagger
        pos_tag.remove([''])
        lista=[]
        for token in pos_tag:
            palavra = pre_processing_text(token[0])
            lista.append(palavra)
            
        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        frase_polarity = lexico_sentimento_SentWordNetPT(lista)
        
        #REALIZAR AVALIAÇÃO COM SENTILEX
        #frase_polarity = lexico_sentimento_SentiLex(lista)
        
        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(lista)
        
        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(lista)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        #frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', lista)
        
        
        
        #print("\nPALAVRAS COM POLARIDADE:\n",frase_polarity)

        #realizar marcação de termos da lista
        words = []
        for token in pos_tag:
            termo = [token[0],token[1]]
            words.append(termo)

        
        #print("\n",words)
        
        frase_polarity2 = frase_polarity

        existe = False
        #pega cada palavra do review
        for i,termo in enumerate(words):
            apont = i
            anterior=words[apont-1]
            
            #busca termos que são substantivos
            
            if (token[1] == 'NCMS') or (token[1] == 'NCFS') or (token[1] == 'NCFP') or (token[1] == 'NCCP') or (token[1] == 'NCCS') or (token[1] == 'NCCI'):
                x=i+2
                posterior = words[i+1:x] #fatia a lista para pegar termo posterior
                #print("Verificou substantivo\n")
                post = ''
                for wrd,ps in posterior:
                    post = wrd
                
                if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                    palavra = termo[0]
                    #busca palavra na lista que contém polaridade
                    for j,item in enumerate(frase_polarity):
                        
                        #print("Entrou negação")
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        
                        for wd,p in poeio:
                            pt = wd
                        #se palavra de negação existe, atribui polaridade negativa   
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                
                if post in negacao: #busca se termo anterior é palavra de negação
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wd,p in poeio:
                                pt = wd
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0]:
                                item[1] = '-1'

                                
                if(anterior[1]=='AQ0' or anterior[1]=='AQA' or anterior[1]=='QAC'  or anterior[1]=='AQS' or anterior[1]=='AO0' or anterior[1]=='AOA' or anterior[1]=='AOC' or  anterior[1]=='AOS'): #verifica se termo anterior é adjetivo
                    #print(anterior[0])
                    #print("Entrou Adjetivo antes\n")
                    existe = True
                    palavra = termo[0]
                    ant = anterior[0]
                    #print(ant)

                    #busca palavra na lista que contém polaridade
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wd,p in poeio:
                            pt = wd
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("Pegou")
                        #atribui polaridade do adjetivo ao substantivo    
                        if palavra == item[0]:
                            #print("Atribuiu",polaridade_adj)
                            item[1] = polaridade_adj
                            
                            
                       
                
                for wrd,ps in posterior:
                    
                    #verifica se termo posterior é adjetivo
                    if(ps == 'AQ0' or ps=='AQA' or ps=='QAC'  or ps=='AQS' or ps=='AO0' or ps=='AOA' or ps=='AOC' or  ps=='AOS'):
                        existe = True
                        polaridade_adj_pos=0
                        #print("Entrou adjetivo depois")
                        #print(wrd)
                                                
                        palavra = termo[0]
                        ant = anterior[0]

                        #busca palavra na lista que contém polaridade 
                        for k,item in enumerate(frase_polarity):
                            ap = k
                            
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                            
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                
                                
                            #pega polaridade do adjetivo
                            if wrd == item[0]:
                                polaridade_adj_pos = item[1]
                                #print("Pegou")
                                #if frase_polarity[k-1]

                            #atribui polaridade do adjetivo ao substantivo      
                            for g,item in enumerate(frase_polarity):
                                if palavra == item[0] and ant == antes:
                                    #print("Atribuiu depois",polaridade_adj_pos)
                                    item[1] = polaridade_adj_pos
                        
        if existe is False: #se não foi avaliado pela técnica
            #print("não existe adjetivo depois de substantivo!")
            frase_polarity = frase_polarity2 #realiza apenas soma das polaridades das palavras
            #print("trocou frase com polaridades:\n", frase_polarity2)
            
                
        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
        polaridade_rev = 0
        #soma polaridades do review após aplicar técnica
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
        
        print("\n",polaridade_rev)

        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)

        print("REVIEW Nº:\t",cont)
            
    acertos = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    zeros = 0
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i] and result_review[i] == 1.0:
            TP += 1

        if int(polarity[1]) == result_review[i] and result_review[i] == -1.0:
            TN += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == 1.0:
            FP += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == -1.0:
            FN += 1
   
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
            
        if int(polarity[1]) == 0 or  result_review[i] == 0.0:
            
            zeros += 1 #conta reviews que deram 0   
        else:
            print("")
    
    print("TP: ",TP,"\tTN: ",TN,"\tFP: ",FP,"\tFN: ",FN)            
    print("TOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    #acuracia = acertos/(len(all_reviews))*100
    #print("\n\n\n\n\nacuracia:\t",acuracia,"%")
    avaliacao(TP, TN, FP, FN, acertos)


"""
def tec_aspecto_adjetivo_TreeTagger(all_reviews):

    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    

    with open(os.path.join("USO_GERAL1.p"), "rb") as file: #-.Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
        
    result_review = []

    
    
    #CHAMA DICIONARIO DO SENTILEX Processed_Reviews_polarity
    #print("DICIONARIO SENTILEX: \n",Sentilex())
    sent_words = Sentilex()
    cont = 0
    
    #pega um review de cada vez
    for i,review in enumerate(all_reviews):

        pos_tag = TreeTagger(review) #passo um review por parametro para a função tretagger
        pos_tag.remove([''])
        lista=[]
        for token in pos_tag:
            palavra = pre_processing_text(token[0])
            lista.append(palavra)


        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        frase_polarity = lexico_sentimento_SentWordNetPT(lista)
        
        #REALIZAR AVALIAÇÃO COM SENTILEX
        #frase_polarity = lexico_sentimento_SentiLex(lista)
        
        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(lista)
        
        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(lista)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        #frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', lista)
        
        
        
        #print("\nPALAVRAS COM POLARIDADE:\n",frase_polarity)

        #realizar marcação de termos da lista
        words =[]
        for token in pos_tag:
            termo = [token[0],token[1]]
            words.append(termo)

        
        #print("\n",words)
        frase_polarity2 = frase_polarity

        existe = False
        #pega cada palavra do review
        for i,termo in enumerate(words):
            apont = i
            anterior=words[apont-1]
            
            #busca termos que são substantivos
            
            if (token[1] == 'NCMS') or (token[1] == 'NCFS') or (token[1] == 'NCFP') or (token[1] == 'NCCP') or (token[1] == 'NCCS') or (token[1] == 'NCCI'):
                x=i+2
                posterior = words[i+1:x] #fatia a lista para pegar termo posterior
                #print("Verificou substantivo\n")
                post = ''
                
                for wrd,ps in posterior:
                    post = wrd
                #print(post)
                if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                    palavra = termo[0]
                    #busca palavra na lista que contém polaridade
                    for j,item in enumerate(frase_polarity):
                        
                        #print("Entrou negação")
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        pt = ''
                        for wd,p in poeio:
                            pt = wd
                        #se palavra de negação existe, atribui polaridade negativa   
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                #print(post)
                #busca se termo posterior é palavra de negação
                if post in negacao: 
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wd,p in poeio:
                                pt = wd
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0]:
                                item[1] = '-1'       
                            
                if(anterior[1]=='AQ0' or anterior[1]=='AQA' or anterior[1]=='QAC'  or anterior[1]=='AQS' or anterior[1]=='AO0' or anterior[1]=='AOA' or anterior[1]=='AOC' or  anterior[1]=='AOS'): #verifica se termo anterior é adjetivo
                    #print(anterior[0])
                    #print("Entrou Adjetivo antes\n")
                    existe = True
                    palavra = termo[0]
                    ant = anterior[0]
                    #print(ant)

                    #busca palavra na lista que contém polaridade
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wd,p in poeio:
                            pt = wd
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("Pegou")
                        #atribui polaridade do adjetivo ao substantivo    
                        if palavra == item[0]:
                            #print("Atribuiu",polaridade_adj)
                            item[1] = polaridade_adj
                            
                            
                       
                
                for wrd,ps in posterior:
                    
                    #verifica se termo posterior é adjetivo
                    if(ps == 'AQ0' or ps=='AQA' or ps=='QAC'  or ps=='AQS' or ps=='AO0' or ps=='AOA' or ps=='AOC' or  ps=='AOS'):
                        existe = True
                        polaridade_adj_pos=0
                        #print("Entrou adjetivo depois")
                        #print(wrd)
                                                
                        palavra = termo[0]
                        ant = anterior[0]

                        #busca palavra na lista que contém polaridade 
                        for k,item in enumerate(frase_polarity):
                            ap = k
                            
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                            
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                
                                
                            #pega polaridade do adjetivo
                            if wrd == item[0]:
                                polaridade_adj_pos = item[1]
                                #print("Pegou")
                                #if frase_polarity[k-1]

                            #atribui polaridade do adjetivo ao substantivo      
                            for g,item in enumerate(frase_polarity):
                                if palavra == item[0] and ant == antes:
                                    #print("Atribuiu depois",polaridade_adj_pos)
                                    item[1] = polaridade_adj_pos


        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
        polaridade_rev = 0
        adj_tem_polaridade = False
        pol_adj = []
        #print(existe)
        if existe is False: #se não foi avaliado por aspecto buscará polaridade dos adjetivos
            frase_polarity = frase_polarity2
            #print("verificou que não tem substantivo e entrou no primeiro False")
            for i,termo in enumerate(words):
                
                apont = i
                anterior=words[apont-1]
                
                #busca termos que são adjetivos
                
                if(termo[1] == 'AQ0' or termo[1]=='AQA' or termo[1]=='QAC'  or termo[1]=='AQS' or termo[1]=='AO0' or termo[1]=='AOA' or termo[1]=='AOC' or  termo[1]=='AOS'):
                    x=i+2
                    posterior = words[i+1:x] #fatia a lista para pegar termo posterior
                    #print("Verificou adjetivo\n")
                    if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wd,p in poeio:
                                pt = wd
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0] and post == pt:
                                item[1] = '-1'
                            
                    if post in negacao: #busca se termo anterior é palavra de negação
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wd,p in poeio:
                                pt = wd
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0]:
                                item[1] = '-1'
                                
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wd,p in poeio:
                            pt = wd
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if termo[0] == item[0]:
                            polaridade_adj = item[1]
                            if polaridade_adj != '0':
                                adj_tem_polaridade = True
                                pol_adj.append(polaridade_adj)
                    
        
                    
        
        
            if adj_tem_polaridade is True:
                #print("tem adjetivo com polaridade!\n")
                for i in pol_adj:
                    polaridade_rev = int(i) + int(polaridade_rev)
                
                #print("\n",polaridade_rev)

                #busca se resultado da soma torna review positivo
                if polaridade_rev >= 1:
                    polaridade_rev = 1
                    cont +=1
                    result_review.append(1)

                #busca se resultado da soma torna review negativo    
                if polaridade_rev <= -1:
                    polaridade_rev = -1
                    cont +=1
                    result_review.append(-1)

                #busca se resultado da soma torna review neutro    
                if polaridade_rev == 0:
                    polaridade_rev = 0
                    cont +=1
                    result_review.append(0)

                
                
            elif adj_tem_polaridade is False: #se não foi avaliado pela técnica
                #print("não tem adjetivo com polaridade!")
                frase_polarity = frase_polarity2 #realiza apenas soma das polaridades das palavras
                #print("trocou frase com polaridades:\n", frase_polarity2)
                polaridade_rev = 0
                #soma polaridades do review após aplicar técnica
                for item,pol in frase_polarity:
                    valor = int(pol)
                    polaridade_rev = polaridade_rev + valor

                print("REVIEW Nº:\t",cont)
                print("\n",polaridade_rev)

                #busca se resultado da soma torna review positivo
                if polaridade_rev >= 1:
                    polaridade_rev = 1
                    cont +=1
                    result_review.append(1)

                #busca se resultado da soma torna review negativo    
                if polaridade_rev <= -1:
                    polaridade_rev = -1
                    cont +=1
                    result_review.append(-1)

                #busca se resultado da soma torna review neutro    
                if polaridade_rev == 0:
                    polaridade_rev = 0
                    cont +=1
                    result_review.append(0)

    if existe is True:
        polaridade_rev = 0
        #soma polaridades do review após aplicar técnica
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor

        print("REVIEW Nº:\t",cont)        
        print("\n",polaridade_rev)

        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)        

        
        print(result_review)
        
    acertos = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        
        print(polarity[1])
        print("\n")
        if int(polarity[1]) == result_review[i] and result_review[i] == 1.0:
            TP += 1

        if int(polarity[1]) == result_review[i] and result_review[i] == -1.0:
            TN += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == 1.0 and int(polarity[1]) == -1.0:
            FP += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == -1.0 and int(polarity[1]) == 1.0:
            FN += 1
   
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
        else:
            print("")
    
    print("TP: ",TP,"\tTN: ",TN,"\tFP: ",FP,"\tFN: ",FN)            
    print("TOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    #acuracia = acertos/(len(all_reviews))*100
    #print("\n\n\n\n\nacuracia:\t",acuracia,"%")
    avaliacao(TP, TN, FP, FN, acertos)

"""   
            
def tec_aspecto_palavra_Palavras(all_reviews):
    all_tokenized_reviews = []
    #palavras de negação para utilizar em técnica
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco', 'mal'] #mal
    
    
    with open(os.path.join("Processed_Reviews_polarity.p"), "rb") as file:  #Processed_Reviews_polarity
        polarity_reviews = pickle.load(file)
    result_review = []
    
    #faz chamada da biblioteca spacy e atribui a uma variável
    spc = spacy.load('pt_core_news_sm')
    tratados = []

    
    
    sent_words = Sentilex()
    cont = 0

    
    #verifica cada review
    for i,review in enumerate(all_reviews):
        #print("REVIEW: \n",review)
        
        lista=[]
        for token in review:
            
            palavra = pre_processing_text(token[1])
            lista.append(palavra)

        #REALIZAR AVALIAÇÃO COM SENTWORDNET-PT-BR
        #frase_polarity = lexico_sentimento_SentWordNetPT(lista)
                    
                    
        #REALIZAR AVALIAÇÃO COM SENTILEX
        #frase_polarity = lexico_sentimento_SentiLex(lista)

        #REALIZAR AVALIAÇÃO COM LIWC
        #frase_polarity = lexico_sentimento_LIWC(lista)

        #REALIZAR AVALIAÇÃO COM OpLexicon
        #frase_polarity = lexico_sentimento_OpLexicon(lista)

        #REALIZAR AVALIAÇÃO COM LÉXICOS CONCATENADOS
        frase_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex', lista)
        
        #print(frase_polarity)
        #print("\n")
        #print("REVIEW COM POLARIDADE: \n",frase_polarity)
        #print("\n")
        
        frase_polarity2 = frase_polarity
        
        tagger = []
        wd = []

        #realiza marcação de cada palavra
        for tok in review:
            palavr = pre_processing_text(tok[1])
            termo = [palavr,tok[3]]
            wd.append(termo)

        #print(wd)
        #print("\n")
        existe = False
        #pega cada palavra do review
        for i,termo in enumerate(wd):
            apont = i
            #print(i)
            
            if i < len(wd):
                anterior=wd[apont-1]
                
            #print(anterior)
            #busca termos que são substantivos
            
            if(termo[1] == 'n' ):
                x=i+2
                posterior = wd[i+1:x] #fatia a lista para pegar termo posterior
                #print("Verificou substantivo\n")
                post = ''
                
                
                #print(posterior)
                
                for token in posterior:
                    post = token[0]
                
                    
                #print(post)
                
                if anterior[0] in negacao: #busca se termo anterior é palavra de negação
                    palavra = termo[0]
                    #busca palavra na lista que contém polaridade
                    for j,item in enumerate(frase_polarity):
                        
                        #print("Entrou negação")
                        ap = j
                        y = j+2
                        
                        poeio=frase_polarity[j+1:y]
                        
                        for wordss,p in poeio:
                            pt = wordss
                            
                        #se palavra de negação existe, atribui polaridade negativa   
                        if palavra == item[0] and post == pt:
                            item[1] = '-1'
                            
                
                if post in negacao: #busca se termo anterior é palavra de negação
                        palavra = termo[0]
                        #busca palavra na lista que contém polaridade
                        for j,item in enumerate(frase_polarity):
                            
                            #print("Entrou negação")
                            ap = j
                            y = j+2
                            
                            poeio=frase_polarity[j+1:y]
                            
                            for wordss,p in poeio:
                                pt = wordss
                                
                            #se palavra de negação existe, atribui polaridade negativa   
                            if palavra == item[0]:
                                item[1] = '-1'

                #print(anterior)                
                if(anterior[1]=='adj'): #verifica se termo anterior é adjetivo
                    #print(anterior[0])
                    #print("Entrou Adjetivo antes\n")
                    existe = True
                    palavra = termo[0]
                    ant = anterior[0]
                    #print(ant)

                    #busca palavra na lista que contém polaridade
                    for k,item in enumerate(frase_polarity):
                        ap = k
                        
                        z = k+2
                        
                        poeio=frase_polarity[k+1:z]
                        
                        for wordss,p in poeio:
                            pt = wordss
                            polaridade_adj=p

                        #pega polaridade do adjetivo    
                        if ant == item[0]:
                            polaridade_adj = item[1]
                            #print("Pegou")
                        #atribui polaridade do adjetivo ao substantivo    
                        if palavra == item[0]:
                            #print("Atribuiu",polaridade_adj)
                            item[1] = polaridade_adj
                            
                            
                       
                
                for wrd,ps in posterior:
                    
                    #verifica se termo posterior é adjetivo
                    if(ps == 'adj'):
                        existe = True
                        polaridade_adj_pos=0
                        #print("Entrou adjetivo depois")
                        #print(wrd)
                                                
                        palavra = termo[0]
                        ant = anterior[0]

                        #busca palavra na lista que contém polaridade 
                        for k,item in enumerate(frase_polarity):
                            ap = k
                            
                            z = k+2

                            antes, tg = frase_polarity[k-1]

                            post_polar = frase_polarity[k+1:z]
                            
                            for a,b in post_polar:
                                polaridade_adj_pos = b
                                
                                
                            #pega polaridade do adjetivo
                            if wrd == item[0]:
                                polaridade_adj_pos = item[1]
                                #print("Pegou")
                                #if frase_polarity[k-1]

                            #atribui polaridade do adjetivo ao substantivo      
                            for g,item in enumerate(frase_polarity):
                                if palavra == item[0] and ant == antes:
                                    #print("Atribuiu depois",polaridade_adj_pos)
                                    item[1] = polaridade_adj_pos
                        
        if existe is False: #se não foi avaliado pela técnica
            #print("não existe adjetivo depois de substantivo!")
            frase_polarity = frase_polarity2 #realiza apenas soma das polaridades das palavras
            #print("trocou frase com polaridades:\n", frase_polarity2)                           
              
        #print("\nFRASE COM POLARIDADE PÓS TÉCNICA:\n",frase_polarity)
         
        polaridade_rev = 0
        #soma polaridades do review após aplicar técnica
        for item,pol in frase_polarity:
            valor = int(pol)
            polaridade_rev = polaridade_rev + valor
        
        print("\n",polaridade_rev,"\n")
        #busca se resultado da soma torna review positivo
        if polaridade_rev >= 1:
            polaridade_rev = 1
            cont +=1
            result_review.append(1)

        #busca se resultado da soma torna review negativo    
        if polaridade_rev <= -1:
            polaridade_rev = -1
            cont +=1
            result_review.append(-1)

        #busca se resultado da soma torna review neutro    
        if polaridade_rev == 0:
            polaridade_rev = 0
            cont +=1
            result_review.append(0)

        print("REVIEW Nº:\t",cont)
            
    acertos = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    zeros = 0
    #busca reviews com polaridade atribuido (de 0 a 5) e compara com resultado da técnica
    for i,polarity in enumerate(polarity_reviews):
        #print(polarity)
        print("\n")
        if int(polarity[1]) == result_review[i] and result_review[i] == 1.0:
            TP += 1

        if int(polarity[1]) == result_review[i] and result_review[i] == -1.0:
            TN += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == 1.0:
            FP += 1

        if int(polarity[1]) != result_review[i] and result_review[i] == -1.0:
            FN += 1
   
        if int(polarity[1]) == result_review[i]:
            acertos += 1 #conta acertos
            
        if int(polarity[1]) == 0 or  result_review[i] == 0.0:
            
            zeros += 1 #conta reviews que deram 0   
        else:
            print("")
    
    print("TP: ",TP,"\tTN: ",TN,"\tFP: ",FP,"\tFN: ",FN)            
    print("TOTAL REVIEWS AVALIADOS:\t",cont)
    print("total de reviews com polaridade:\t",len(all_reviews))
    print("ACERTOS:\t",acertos)
    #realiza acurácia
    #acuracia = acertos/(len(all_reviews))*100
    #print("\n\n\n\n\nacuracia:\t",acuracia,"%")
    avaliacao(TP, TN, FP, FN, acertos)


all_reviews = []
"""
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/lexico"):
    for filename in fnmatch.filter(files, '*.txt'):
            f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
            review = f.read()
            print(review)
            review = pre_processing_text(review, use_normalizer=True)
            all_reviews.append(review)
    with open("tec_linha_de_base.p", "wb") as f:
        pickle.dump(all_reviews, f) Processed_Reviews 
"""

with open(os.path.join("Corpus Buscape/Analisados/salvando_teste/analisados_processados.p"), "rb") as file: #->Processed_Reviews
        all_reviews = pickle.load(file)

tec_aspecto_palavra_Palavras(all_reviews)
#tec_aspecto_palavra_spacy


