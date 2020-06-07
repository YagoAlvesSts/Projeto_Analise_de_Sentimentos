#encoding: utf-8
import os    
"""
#lista apenas os arquivos txt da pasta
pasta = "C:/Users/yagoa/Desktop/Faculdade/PP/ultimos/Corpus Buscape/reviews/Celular e Smartphone"
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]    
arquivos_txt = [arq for arq in arquivos if arq.lower().endswith(".txt")]

#cria uma lista para armazenar as saídas
saida = []    

#percorre os arquivos
for arq in arquivos_txt:    
  #abre o arquivo 
  with open(arq) as f:
      linhas = f.readlines()

  #soma os valores
  soma = 0
  for linha in linhas:
     soma += int(linha.split(" ")[0].strip())

  #guarda na lista de saida  
  saida.append("O total por segundo no arquivo {} é: {} \n".format(arquivos_txt,soma))

#grava a lista em um novo arquivo
arq_saida = open('/arq_saida.txt', 'w')
arq_saida.writelines(saida)
arq_saida.close()


"""
from pathlib import Path


path = Path("C:/Users/yagoa/Desktop/Faculdade/PP/ultimos/Corpus Buscape/Celular e Smartphone")
letters_files = path.glob('*_*.txt')
letters = [letter.read_text(encoding = 'utf-8') for letter in sorted(letters_files)]

print(letters)



# Criando e escrevendo em arquivos de texto (modo 'w').
arquivo = open('matriz01.txt','w')
arquivo.writelines(letters)
arquivo.close()


# Lendo o arquivo criado:
arquivo = open('matriz01.txt','r')
texto=arquivo.readlines()
texto = texto.replace("\n")
print(texto)
arquivo.close()

