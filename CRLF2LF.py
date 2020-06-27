import os
import pickle

#Dessa forma preciso refazer o Processed_Reviews 
with(open(os.path.join("Processed_Reviews.p"), "rb")) as file:
    corpus = pickle.load(file)    

for i, review in enumerate(corpus):
    text = ""
    for letra in review:
        if letra == ".":
            text = text+letra+"\n"
            
        else:
            text = text+letra

    if letra != ".":
        text = text+".\n"

    sentencas = text.split("\n")
    text = ""
    for x, sentenca in enumerate(sentencas): #essa função eu fiz pra tirar o espaço do começo da frase
        sentenca = sentenca.strip() 
        text = text+sentenca+"\n"        
        
    file =  open(os.path.join("Corpus Buscape/for_palavras", "id_"+str(i)+".txt"), "w", encoding="utf8" )
    file.writelines(text[:-1])    #esse -1 eu coloquei pq a última posição era \n a mais mas se quiser pode tirar o -1 
    file.close()

#apartir daqui é pra converter de crlf para lf
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



