import os
import pickle
from ftfy import fix_encoding
import subprocess
from collections import Counter

def most_commom(lst):
    #length = len(lst)

    #porc = ((length * 3)/100)
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

def pre_processing_text(text):
    str(text)
    text = text.lower()

    input_chars = ["\n", ".", "!", "?", "ç", " / ", " - ", "|", "ã", "õ", "á", "é", "í", "ó", "ú", "â", "ê", "î", "ô", "û", "à", "è", "ì", "ò", "ù"]
    output_chars = [" . ", " . ", " . ", " . ", "c", "/", "-", "", "a", "o", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u"]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text

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


def aspecto_substantivo_TreeTagger(save = "False"):

    with open(os.path.join("Processed_Reviews.p"), "rb") as file: #->Processed_Reviews
        sent_words = pickle.load(file)
                
    all_reviews = sent_words
    polaridade  = sent_words
    reviews = []
    for review in all_reviews:
        #print(review)
        #review = pre_processamento(review)
        reviews.append(review)# pre_processamento(review) acho q nem precisa desse pré processamento
        all_reviews = reviews
    with open("Nounprocessed_Reviews_TreeTagger.p", "wb") as f:
        pickle.dump(all_reviews, f)
    

    
    mais_comum = []
    subst = []
    aspects = []
    #COMEÇAAAAAAAAAAAAAAAAAAAA AQUIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
    for i, text in enumerate(all_reviews[:2000]):
        #print(i)
        pos_tag = TreeTagger(text) #passo um review por parametro para a função tretagger
        pos_tag.remove([''])
        for token in pos_tag:
            #token[0] é a palavra, token[1] a tag e token[3] é a palavra lematizada
            if (token[1] == 'NCMS') or (token[1] == 'NCFS') or (token[1] == 'NCFP') or (token[1] == 'NCCP') or (token[1] == 'NCCS') or (token[1] == 'NCCI'):
                #TERMINAAAAAAAAAAAAAAAAAAAA AQUIIIIIIIIIIIIIIIIIIIIIIIIIIIII :)
                palavra = pre_processing_text(token[0])
                subst.append(str(palavra))

    mais_comum = most_commom(subst)
    #print("\n",mais_comum[:200])
    
    for tupla in mais_comum:
        aspects.append(tupla[0])
    
    #print(aspects[:200])
    print("PRONTIN !!!")
    if save:

        
        with open(os.path.join("Aspectos","noun_aspects_TreeTagger.p"), "wb") as f:
            pickle.dump(aspects, f)
        arquivo = open('Aspectos/noun_aspects_TreeTagger.txt','w')
        arquivo.write(' '.join(aspects))
        arquivo.close()
        f = open('Aspectos/all_noun_aspects_TreeTagger.txt','w')
        f.write(' '.join(aspects))
        f.close()



    return aspects

aspecto_substantivo_TreeTagger()
