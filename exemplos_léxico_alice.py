def polaridade_comentarios():
    stop_words = stopwords.words('portuguese')    
    stop_words.extend(['ser','estar'])
    result_review = []
    polaridade_comentario = []
    negacao = ['jamais','nada','nem','nenhum','ninguém','nunca','não','tampouco', 'mal'] #mal
    #reviews, polarity_reviews = comentarios('train')
    with open(os.path.join("Corpus_reduzido", "corpus.p"), "rb") as file:
        reviews = pickle.load(file)      

    with open(os.path.join("Corpus_reduzido", "corpus_polaridade.p"), "rb") as file:
        polarity_reviews = pickle.load(file)
    sent_words, sent_words_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex')
    #sent_words, sent_words_polarity = lexicos_sentimento_OpLexicon()
    
    for i, review in enumerate (reviews[:2000]):
        #review = review.split()
        print(i)
        palavra = []
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(review)     
        palavra = ['','','']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
        
        for x, text in enumerate (doc):
            x = x + 3
            palavra.append(str(text))
            word = (str(text.lemma_))
                   
            if text.pos_ == 'ADJ':
                word = (str(text))    
                
            if word in sent_words and word not in stop_words:
                troca = False
                
                if palavra[x-1] in negacao or palavra[x-2] in negacao or palavra[x-3] in negacao:
                    troca = True                        
                    
                if sent_words_polarity[word] == '1':
                    if troca:
                        #print('troca')
                        polaridade += -1
                    else:    
                        polaridade += 1
                    #print(word)
                    #print(sent_words_polarity[word])
                        
                elif sent_words_polarity[word] == '-1':
                    if troca:
                        #print('troca')
                        polaridade += 1
                    else:
                        polaridade += -1
                    #print(word)
                    #print(sent_words_polarity[word])

        if polaridade >= 0:
            result_review.append(1)
        else:
            result_review.append(-1)

        #print(review)
        #print(result_review[i])
        #print(polaridade)
        
    acertos = 0
    for i, pol in enumerate(polaridade_comentario):
        if pol == result_review[i]:
            acertos += 1
        else:
            print(reviews[i])
            
    print(acertos)
    print(len(polaridade_comentario))
    acuracia = acertos/(len(polaridade_comentario))*100
    print("acuracia: ", acuracia,"%")





























def polaridade_comentarios_2():
    stop_words = stopwords.words('portuguese')
    stop_words.extend(['ser','estar'])
    negacao = ['jamais','nada','nem','nenhum','ninguém','nunca','não','tampouco', 'mal']
    intensificacao = ['mais','muito','demais','completamente','absolutamente','totalmente','definitivamente','extremamente','frequentemente','bastante']
    reducao = ['pouco','quase','menos','apenas']
    palavra = []
    result_review = []
    polaridade_comentario = []
    
    reviews, polarity_reviews = comentarios('train')
    #sent_words, sent_words_polarity = lexicos_sentimento_LIWC()
    sent_words, sent_words_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex')
    aux1 = list()
    aux2 = list()
    dados = list()
    aux1 = reviews(len[reviews-1000:])
    aux2 = reviews[:1000]
    dados = aux1 + aux2
    for i, review in enumerate (dados):
        #review = review.split()
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(review)       
        palavra = ['','','']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[review])
        print(i)
        
        for x, text in enumerate (doc):
            x = x + 3
            palavra.append((str(text)))
            word = (str(text.lemma_))
            
            '''if text.pos_ == 'ADJ':
                word = (str(text))'''
                
            if word in sent_words and word not in stop_words:
                polaridade = float(sent_words_polarity[word])
                #print(word)
                #print(polaridade)
                
                if palavra[x-1] in intensificacao or palavra[x-2] in intensificacao or palavra[x-3] in intensificacao:
                    if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao:
                        polaridade = polaridade/3                        
                    else:
                        polaridade = polaridade*3

                elif palavra[x-1] in reducao or palavra[x-2] in reducao or palavra[x-3] in reducao:
                    if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao:
                        polaridade = polaridade*3
                    else:
                        polaridade = polaridade/3
                        
                elif palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao:
                    polaridade = -1*polaridade
   
                sentimento_texto = sentimento_texto + polaridade
                
        if sentimento_texto >= 0:
            result_review.append(1)
        else:
            result_review.append(-1)
                
    acertos = 0
    for i, pol in enumerate(polaridade_comentario):
        if pol == result_review[i]:
            acertos += 1
        else:
            print(reviews[i])
        
    print(acertos)
    print(len(polaridade_comentario))
    acuracia = acertos/(len(polaridade_comentario))*100
    print("acuracia: ", acuracia,"%")    
