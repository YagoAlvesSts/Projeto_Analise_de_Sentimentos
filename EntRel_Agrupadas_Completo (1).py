# coding: latin1 
"""
Created on Thu Jan  2 10:00:06 2014

@author: Marcio Dias
"""

import os
import nltk
import xml.etree.ElementTree as ET

# Grade = [ [ '-' for j in range(len(ListaAbertasRadical)+1) ] for i in range(len(VecSent)+1) ] 

Rel = []     #vetor de relações
Seg = []     #vetor de seguimentos
Seg_id = []
Nuc_Sat_Seg = []  # Informação núcleo/satélite do seguimento
Eh_Sent = []      #Encontrei um ponto final que delimita uma sentença? (Y/N)
Nuc_Sat_Rel = []  #Informação núcleo/satélite da relação 
VecSpan = []     # Vetor de Span (palavras ou techos de texto)
VecID = []      #Vetor de identificador das sentenças 
VecSpan2 = []
VecSentS=[]
MatrizRST = []
#D2_C1_Estadao_04-08-2006_05h54.txt.seg
VecSent = []
VecIndSent =[]
Mais_Nuc = []     #Seguimento mais nuclear
ListAtribPal = [[]]
ListaAbertasRadical = []
ListaAbertasRadical2 = []
ListaPalSent =[[[]]]
ListaPalavrasAbertas = []
ListaPalavrasAbertas2 = []
VecSpan = []     # Vetor de Span (palavras ou techos de texto)
VecID = []      #Vetor de identificador das sentenças 
VecSpan2 = []
VecSentRadical = []
marcador = []
filho = 0
Entidades = []
Entidades2 = []


    
def Salvar(nomeArq,Grade1):                       
    arq = open(nomeArq, 'w')
    for i in range(len(Grade1)):
        for j in range(len(Grade1[i])):
            arq.write(str(Grade1[i][j]).encode('utf8'))
            arq.write(' ')
        if i != len(Grade1) - 1:    
            arq.write(str('\n'))
    arq.close()


def CriarMatrizRST(size):
    a = []
    for i in range(size):
        for j in range(size):
            a.append('-')
        MatrizRST.append(a)
        a = []
        
def LerArqSents(NomeArq):
    a=open(NomeArq,'r')
    for linha in a:
        frase = str(linha)
        VecSent.append(frase.lower())
        VecSentS.append(linha)
    CriarMatrizRST(len(VecSentS))
    
def PreencherMatrizRST(NoXML,Grade):
    if NoXML.tag=='edu':
        marcador.append(NoXML.tag)
        Nuc_Sat_Seg.append(NoXML.attrib["status"])
        Seg.append(NoXML.text)
        #conta = 0
        
        seguimento = str(NoXML.text)
        for i in range(len(VecSent)):
            if seguimento in VecSentS[i]:
                #if conta == 0:
                    #conta = conta + 1
                for j in range(len(Entidades2)):
                    if Entidades2[j] in seguimento.lower():
                        tam = len(Rel)-1
                        for x in range(tam, -1, -1): 
                            if Grade[i+1][j+1] == '-':                            
                                Grade[i+1][j+1] = Rel[x]+ "*"
                            elif x == tam:
                                x = x - 1
                            else:
                                rel = Rel[x]
                                if rel in Grade[i+1][j+1]:
                                    print "---"                                                                        
                                else:
                                    Grade[i+1][j+1] = Grade[i+1][j+1]+Rel[x]+"*"
                            '''
                            if len(ListaAbertasRadical2[j]) <= 3:  #radicais de tamanho 3 ou menor :
                                    
                                            palavra = ''
                                            loc = seguimento.find(ListaAbertasRadical2[j])
                                            if loc != 0:
                                                if seguimento[loc-1] != ' ' and seguimento[loc-1] != '(':
                                                    palavra = palavra+"&"
                                            while seguimento[loc] != ' ' and seguimento[loc] != ')' and seguimento[loc] != '.' and seguimento[loc] != '?' and seguimento[loc] != '!' and seguimento[loc] != ',' and seguimento[loc] != ']' and seguimento[loc] != '}' and seguimento[loc] != '"' and seguimento[loc] != "'":
                                                if seguimento[loc] != '-':
                                                    palavra = palavra+str(seguimento[loc])
                                                else:
                                                    loc = loc + 1
                                                    palavra = palavra+str(seguimento[loc])
                                                    break
                                                if loc != len(seguimento)-1:
                                                    loc = loc + 1
                                                else:                                                
                                                    break                                            
                                            for u in range(len(ListaAbertasRadical)):
                                                if palavra == ListaPalavrasAbertas2[u]:
                                                    tam = len(Rel)-1
                                                    for x in range(tam, -1, -1): 
                                                        if Grade[i+1][u+1] == '-':                            
                                                            Grade[i+1][u+1] = Rel[x] + "." + Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]+"*"
                                                        else:
                                                            if x == tam:
                                                                rel = Rel[x] + "." + Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]
                                                                if rel in Grade[i+1][u+1]:
                                                                    print "-"                                                                        
                                                                else:
                                                                    Grade[i+1][u+1] = Grade[i+1][u+1]+Rel[x] + "." + Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]+ "*"
                                                            else:
                                                                rel = Rel[x] + "." + Nuc_Sat_Rel[x+1]
                                                                if rel in Grade[i+1][u+1]:
                                                                    print "--"                                                                        
                                                                else:
                                                                    Grade[i+1][u+1] = Grade[i+1][u+1]+Rel[x] + "." + Nuc_Sat_Rel[x+1]+ "*"
                                                    
                            else:
                                tam = len(Rel)-1
                                for x in range(tam, -1, -1): 
                                    if Grade[i+1][j+1] == '-':                            
                                        Grade[i+1][j+1] = Rel[x] + "." + Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]+ "*"
                                    elif x == tam:
                                        x = x - 1
                                    else:
                                        rel = Rel[x] + "." + Nuc_Sat_Rel[x+1]
                                        if rel in Grade[i+1][j+1]:
                                            print "---"                                                                        
                                        else:
                                            Grade[i+1][j+1] = Grade[i+1][j+1]+Rel[x] + "." + Nuc_Sat_Rel[x+1]+ "*"                                                                                                                                                                                                                                                                                                
                            '''                
                #else:
                    #break                            
                                            
                                            
        Seg_id.append(NoXML.attrib["id"])
        Mais_Nuc.append(NoXML.attrib["id"])
    
        if NoXML.text[len(NoXML.text)-1]=='.' or NoXML.text[len(NoXML.text)-2]=='.' or NoXML.text[len(NoXML.text)-1]=='!' or NoXML.text[len(NoXML.text)-2]=='!' or NoXML.text[len(NoXML.text)-1]=='?' or NoXML.text[len(NoXML.text)-2]=='?':
            Eh_Sent.append('Y') 
        else:
            Eh_Sent.append('N')
        return
    else:
        if NoXML.tag!='rst':
            marcador.append(NoXML.tag) 
            if NoXML.attrib["name"] == 'enablement' or NoXML.attrib["name"] == 'motivation':
                nomeRelacao = 'motivation'
            elif NoXML.attrib["name"] == 'evidence' or NoXML.attrib["name"] == 'justify':
                nomeRelacao = 'motivation'
            elif NoXML.attrib["name"] == 'volitional-cause' or NoXML.attrib["name"] == 'non-volitional-cause' or NoXML.attrib["name"] == 'volitional-result' or NoXML.attrib["name"] == 'non-volitional-result' or NoXML.attrib["name"] == 'purpose':
                nomeRelacao = 'cause'     
            elif NoXML.attrib["name"] == 'antithesis' or NoXML.attrib["name"] == 'concession':
                nomeRelacao = 'antithesis'
            elif NoXML.attrib["name"] == 'condition' or NoXML.attrib["name"] == 'otherwise':
                nomeRelacao = 'condition'
            elif NoXML.attrib["name"] == 'interpretation' or NoXML.attrib["name"] == 'evaluation':
                nomeRelacao = 'interpretation'
            elif NoXML.attrib["name"] == 'restatement' or NoXML.attrib["name"] == 'summary':
                nomeRelacao = 'summary'
            elif NoXML.attrib["name"] == 'sequence' or NoXML.attrib["name"] == 'contrast':
                nomeRelacao = 'other_relations'
            else:
                nomeRelacao = NoXML.attrib["name"]
            Rel.append(nomeRelacao)
            Nuc_Sat_Rel.append(NoXML.attrib["status"])
        for child in NoXML.getchildren():
            PreencherMatrizRST(child,Grade)
        if Eh_Sent[len(Eh_Sent)-1]=='Y'and Eh_Sent[len(Eh_Sent)-2]=='Y':
            if (Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]=='Nuc'and Nuc_Sat_Seg[len(Nuc_Sat_Seg)-2]=='Nuc') or (Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]=='Nuc'and Nuc_Sat_Seg[len(Nuc_Sat_Seg)-2]=='Sat') or (Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]=='Sat'and Nuc_Sat_Seg[len(Nuc_Sat_Seg)-2]=='Nuc'):
                p1=1
                p2=1
                pto1 = 1
                pto2 = 2
                for i in range(len(VecSentS)):
                    if Seg[len(Seg)-1] in VecSentS[i]:
                        p1=i+1
                        pto1 = i
                        for j in range(len(Entidades)):
                            if len(Rel) == 0:
                                return Grade
                            relacao = Rel[len(Rel)-1]
                            if Entidades2[j] in Seg[len(Seg)-1]:
                                relacoes = Grade[p1][j+1]
                                if relacoes != '-':
                                    if relacao in relacoes:
                                        print "-----"
                                    else:
                                        Grade[p1][j+1] = Grade[p1][j+1]+Rel[len(Rel)-1]+"*"                                                                
                                else:
                                    Grade[p1][j+1] = relacao+'*'
                                '''
                                if len(ListaAbertasRadical2[j]) <= 3:  #radicais de tamanho 3 ou menor :
                                    
                                        palavra = ''
                                        loc = Seg[len(Seg)-1].find(ListaAbertasRadical2[j])
                                        if loc != 0:
                                            if Seg[len(Seg)-1][loc-1] != ' ' and Seg[len(Seg)-1][loc-1] != '(':
                                                palavra = palavra+"&"
                                        while Seg[len(Seg)-1][loc] != ' ' and Seg[len(Seg)-1][loc] != ')' and Seg[len(Seg)-1][loc] != '.' and Seg[len(Seg)-1][loc] != '?' and Seg[len(Seg)-1][loc] != '!' and Seg[len(Seg)-1][loc] != ',' and Seg[len(Seg)-1][loc] != ']' and Seg[len(Seg)-1][loc] != '}' and Seg[len(Seg)-1][loc] != '"' and Seg[len(Seg)-1][loc] != "'":
                                            if Seg[len(Seg)-1][loc] != '-':
                                                palavra = palavra+str(Seg[len(Seg)-1][loc])
                                            else:
                                                loc = loc + 1
                                                palavra = palavra+str(Seg[len(Seg)-1][loc])
                                                break
                                            if Seg[len(Seg)-1][loc] != Seg[len(Seg)-1][-1]:
                                                loc = loc + 1
                                            else:                                                
                                                break                                            
                                        for u in range(len(ListaAbertasRadical)):
                                            if palavra == ListaPalavrasAbertas2[u]:
                                                relacoes = Grade[p1][j+1]
                                                if relacoes != '-':
                                                    if relacao in relacoes:
                                                        print "----"
                                                    else:
                                                        Grade[p1][j+1] = Grade[p1][j+1]+Rel[len(Rel)-1]+"."+Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]+"*"                                                                
                                                else:
                                                    Grade[p1][j+1] = relacao+'*'
                                else:                                
                                     relacoes = Grade[p1][j+1]
                                     if relacoes != '-':
                                         if relacao in relacoes:
                                             print "-----"
                                         else:
                                            Grade[p1][j+1] = Grade[p1][j+1]+Rel[len(Rel)-1]+"."+Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]+"*"                                                                
                                     else:
                                        Grade[p1][j+1] = relacao+'*'
                                '''
                                
                    if Seg[len(Seg)-2] in VecSentS[i]:
                        p2=i+1
                        pto2 = i
                        for j in range(len(Entidades)):
                            relacao = Rel[len(Rel)-1]
                            if Entidades2[j] in Seg[len(Seg)-2]:
                                relacoes = Grade[p2][j+1]
                                if relacoes != '-':
                                    if relacao in relacoes:
                                        print "-------"
                                    else:
                                        Grade[p2][j+1] = Grade[p2][j+1]+Rel[len(Rel)-1]+"*"
                                else:
                                    Grade[p2][j+1] = relacao+'*'                                
                                '''
                                 if len(ListaAbertasRadical2[j]) <= 3:  #radicais de tamanho 3 ou menor :
                                    
                                        palavra = ''
                                        loc = Seg[len(Seg)-2].find(ListaAbertasRadical2[j])
                                        if loc != 0:
                                            if Seg[len(Seg)-2][loc-1] != ' ' and Seg[len(Seg)-2][loc-1] != '(':
                                                palavra = palavra+"&"
                                        while Seg[len(Seg)-2][loc] != ' ' and Seg[len(Seg)-2][loc] != ')' and Seg[len(Seg)-2][loc] != '.' and Seg[len(Seg)-2][loc] != '?' and Seg[len(Seg)-2][loc] != '!' and Seg[len(Seg)-2][loc] != ',' and Seg[len(Seg)-2][loc] != ']' and Seg[len(Seg)-2][loc] != '}' and Seg[len(Seg)-2][loc] != '"' and Seg[len(Seg)-2][loc] != "'":
                                            if Seg[len(Seg)-2][loc] != '-':
                                                palavra = palavra+str(Seg[len(Seg)-2][loc])
                                            else:
                                                loc = loc + 1
                                                palavra = palavra+str(Seg[len(Seg)-2][loc])
                                                break
                                            if Seg[len(Seg)-2][loc] != Seg[len(Seg)-2][-1]:
                                                loc = loc + 1
                                            else:                                                
                                                break                                            
                                        for u in range(len(ListaAbertasRadical)):
                                            if palavra == ListaPalavrasAbertas2[u]:
                                                relacoes = Grade[p2][j+1]
                                                if relacoes != '-':
                                                    if relacao in relacoes:
                                                        print "------"
                                                    else:
                                                        Grade[p2][j+1] = Grade[p2][j+1]+Rel[len(Rel)-1]+"."+Nuc_Sat_Seg[len(Nuc_Sat_Seg)-2]+"*"                                                                
                                                else:
                                                    Grade[p2][j+1] = relacao+'*'
                                                    
                                 else:                                
                                    relacoes = Grade[p2][j+1]
                                    if relacoes != '-':
                                        if relacao in relacoes:
                                            print "-------"
                                        else:
                                            Grade[p2][j+1] = Grade[p2][j+1]+Rel[len(Rel)-1]+"."+Nuc_Sat_Seg[len(Nuc_Sat_Seg)-2]+"*"
                                    else:
                                        Grade[p2][j+1] = relacao+'*'
                                '''
                        
                if len(Rel) == 0:
                    return Grade
                    
                #for j in range(len(ListaAbertasRadical)):
                    #if ListaAbertasRadical2[j] in VecSentS[p1].lower():
                            #Grade[p1+1][j+1] = Grade[p1+1][j+1]+str(Rel[len(Rel)-1])+"."+str(Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1])+'*'
                #for j in range(len(ListaAbertasRadical)):
                    #if ListaAbertasRadical2[j] in VecSentS[p2].lower():
                        #Grade[p2+1][j+1] = Grade[p2+1][j+1]+str(Rel[len(Rel)-1])+"."+str(Nuc_Sat_Seg[len(Nuc_Sat_Seg)-2])+'*'
                        
                                        
                MatrizRST[pto1][pto2]=Rel[len(Rel)-1]
                Rel.pop()
                InfoRel=Nuc_Sat_Rel[len(Nuc_Sat_Rel)-1]
                Nuc_Sat_Rel.pop()
#                q= Seg[len(Seg)-2] + " " + Seg[len(Seg)-1]
                Seg.pop()
#               Seg.pop()
#               Seg.append(q)
                Eh_Sent.pop()
                Mais_Nuc.pop()
                Nuc_Sat_Seg.pop()
                Nuc_Sat_Seg.pop()
                Nuc_Sat_Seg.append(InfoRel)
        else:
            if marcador[len(marcador)-1] == "edu" and marcador[len(marcador)-2] == "edu":
                marcador.pop()
                marcador.pop()
                marcador.pop()
            # nucleo1=Nuc_Sat_Seg[len(Nuc_Sat_Seg)-1]                
            nucleo2=Nuc_Sat_Seg.pop()
            #if len(Nuc_Sat_Seg) == 0:
                    #return Grade
            nucleo1=Nuc_Sat_Seg.pop()
            seg2=Seg.pop()
            seg1=Seg.pop()
            segid2=Seg_id.pop()
            segid1=Seg_id.pop()
            ehsent2=Eh_Sent.pop()
            ehsent1=Eh_Sent.pop()
            Rel.pop()
            Nuc_Sat_Seg.append(Nuc_Sat_Rel.pop())
            Seg_id.append(str(segid1)+"-"+str(segid2))
            Seg.append(str(seg1)+" "+str(seg2))
            Eh_Sent.append(ehsent2)
            Mais_Nuc.pop()
            Mais_Nuc.pop()
            if(nucleo2=='Nuc' and nucleo1=='Sat'):
                Mais_Nuc.append(segid2)
            elif(nucleo1=='Nuc' and nucleo2=='Sat'):
                Mais_Nuc.append(segid1)
            elif(nucleo1=='Nuc' and nucleo2=='Nuc'):
                Mais_Nuc.append(str(segid1)+","+str(segid2))

def LeituraRST(ArqXML,grade,Arq):
    doc = ET.parse(ArqXML)
    node0=doc.getroot()
    node1=node0.find("rst")
    matriz = PreencherMatrizRST(node1,grade)
    Salvar(Arq, matriz)

    
def CriarGradeRel(ArqXML, ArqRel):
    
    stemmer = nltk.stem.RSLPStemmer()
    doc = ET.parse(ArqXML)
    node0=doc.getroot()
    node1=node0.find("corpus")
    node2=node1.find("body")
    for node in node2.getchildren():
        x=node.attrib["id"]
        VecIndSent.append(x)
        node3=node.find("graph")
        for node4 in node3.getchildren():
            if node4.tag == 'terminals':
                for node5 in node4.getchildren():
                    temp=[]
                    temp.append(node5.attrib["id"])
                    temp.append(node5.attrib["word"])
                    temp.append(node5.attrib["lemma"])
                    temp.append(node5.attrib["pos"])
                    temp.append(node5.attrib["morph"])
                    temp.append(node5.attrib["sem"])
                    temp.append(node5.attrib["extra"])
                    ListAtribPal.append(temp)
                ListaPalSent.append(ListAtribPal)
    abertas = ListaPalSent[len(ListaPalSent)-1]
    del abertas[0]
    for i in range(len(abertas)):
        if abertas[i][3] == 'n':
            if Entidades == None:
                Entidades.append(abertas[i][1].lower())
            elif not (abertas[i][1].lower() in Entidades):
                Entidades.append(abertas[i][1].lower())
        if abertas[i][3] == 'prop':
            if Entidades == None:
                Entidades.append(abertas[i][1].lower())
            elif not (abertas[i][1].lower() in Entidades):
                Entidades.append(abertas[i][1].lower())
    
    
    for i in range(len(Entidades)):       #Retirando o _ das palavras abertas reconhecidas pelo PALAVRAS:
        Entidades2.append(Entidades[i].lower())
        if '_' in Entidades[i]:
            word = str(Entidades[i])
            Entidades2[i] = word.replace('_',' ') 
            
    Grade = [ [ '-' for j in range(len(Entidades)+1) ] for i in range(len(VecSent)+1) ] 
  
    for i in range(len(VecSent)):     # linha
        for j in range(len(Entidades)): # coluna
            Grade[0][j+1]= Entidades[j]
        Grade[i+1][0] = i +1
        
        
    doc2 = ET.parse(ArqRel)
    node0=doc2.getroot()
    node1=node0.find("rst")
    node2=node1.find("relation")                
  
    #for a in Grade:
        #print a
    return Grade

x = os.listdir('Textos_Fontes_CSTNews_Seguimentados/')
y = os.listdir('XML_fontes/')
z = os.listdir('RELACOES_NOVAS/')


del x[0]
del y[0]
#del z[0]


for a in range(len(y)):
        
    j = a + 1
#    if a != 0:
    
    w = y[a].find('.')
    w1 = y[a][:w]    
        
           
    print "Para o Arquivo n."+str(j)+": "+w1
    print "PROCESSANDO... AGUARDE!"
    ArqGrade = 'GRADES_COERENTES/'+w1+'_GradeRelRST_COERENTE.txt'
    LerArqSents('Textos_Fontes_CSTNews_Seguimentados/'+x[a])
    grade = CriarGradeRel('XML_fontes/'+y[a], 'RELACOES_NOVAS/'+z[a])
    LeituraRST('RELACOES_NOVAS/'+z[a],grade,ArqGrade)

    if len(Eh_Sent)!=0:
        while len(Eh_Sent) > 0:
            Eh_Sent.pop()    

    if len(ListAtribPal)!=0:
        while len(ListAtribPal) > 0:
            ListAtribPal.pop()

    if len(Entidades)!=0:
        while len(Entidades) > 0:
            Entidades.pop()
            Entidades2.pop()
                
    if len(ListaPalSent)!=0:        
        while len(ListaPalSent) > 0:
            ListaPalSent.pop()
                        
    if len(Mais_Nuc)!=0:
        while len(Mais_Nuc) > 0:
            Mais_Nuc.pop()
            
    if len(MatrizRST)!=0:
        while len(MatrizRST) > 0:
            MatrizRST.pop()
            
    if len(Nuc_Sat_Rel)!=0:            
        while len(Nuc_Sat_Rel) > 0:
            Nuc_Sat_Rel.pop()
            
    if len(Nuc_Sat_Seg)!=0:                
        while len(Nuc_Sat_Seg) > 0:
            Nuc_Sat_Seg.pop()
            
    if len(Rel)!=0:            
        while len(Rel) > 0:
            Rel.pop()
            
    if len(Seg)!=0:            
        while len(Seg) > 0:
            Seg.pop()
            
    if len(Seg_id)!=0:            
        while len(Seg_id) > 0:
            Seg_id.pop()
            
    if len(VecID)!=0:            
        while len(VecID) > 0:
            VecID.pop()
            
    if len(VecIndSent)!=0:            
        while len(VecIndSent) > 0:
            VecIndSent.pop()
            
    if len(VecSent)!=0:            
        while len(VecSent) > 0:
            VecSent.pop()
            
    if len(VecSentRadical)!=0:            
        while len(VecSentRadical) > 0:
            VecSentRadical.pop()
            
    if len(VecSentS)!=0:            
        while len(VecSentS) > 0:
            VecSentS.pop()
            
    if len(VecSpan)!=0:
        while len(VecSpan) > 0:
            VecSpan.pop()

    if len(VecSpan2)!=0:            
        while len(VecSpan2) > 0:
            VecSpan2.pop()
            
    if len(grade)!=0:            
        while len(grade) > 0:
            grade.pop()
            
    if len(marcador)!=0:            
        while len(marcador) > 0:
            marcador.pop()

    if len(VecSpan)!=0:
        while len(VecSpan) > 0:
            VecSpan.pop()
    print "Arquivo n."+str(j)+": Processado \n"
            
print "-- FIM DO PROCESSAMENTO --"


'''

LerArqSents('Textos_Fontes_CSTNews_Seguimentados/D1_C9_Folha_04-08-2006_13h20.txt.seg')
grade = CriarGradeRel('XML_fontes/D1_C9_Folha.txt.xml', 'RELACOES_NOVAS/D1_C9_Folha_04-08-2006_13h20_binaria.rs3.xml')
LeituraRST('RELACOES_NOVAS/D1_C9_Folha_04-08-2006_13h20_binaria.rs3.xml',grade,'marcio.txt')
print "marcio"
'''

