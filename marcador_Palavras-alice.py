import os
import pickle
from ftfy import fix_encoding


def palavras(save="True"):
    count = 0
    analisados = []
    
    while count <= 5:
        '''tive que pegar os arquivos desse jeito pq tem muitos arquivos dai na hora que ele abre ele pula do 0 para o 10000 mas pode abrir do jeito mais chic'''
        endereco = "Corpus Buscape/Analisados/teste/id_" + str(count) + "_Palavras.xml" 
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
        with open(os.path.join("Corpus Buscape/Analisados/salvando_teste","USO_GERAL_analisados_processados.p"), "wb") as f:
            pickle.dump(analisados, f)

    return analisados

palavras(save="True")    
    

            
