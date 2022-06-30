# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:04:07 2022

@author: mau_f
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 22:33:45 2022

@author: mau_f
"""

import json
from datetime import datetime
from TwitterSearch import *
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import tweepy
import configparser


consumer_key= ''
consumer_secret= ''
access_token=''
access_token_secret=''

try:
    ts=TwitterSearch(
        consumer_key=  consumer_key ,
        consumer_secret = consumer_secret  ,
        access_token= access_token ,
        access_token_secret = access_token_secret        
        )
    
    tso=TwitterSearchOrder()
    tso.set_keywords(['BBB'],or_operator = True)
    tso.set_language('pt')
    
    for tweet in ts.search_tweets_iterable(tso):
        print('created_at: ',tweet['created_at'], 'User_id: ',tweet['id_str'],'Tweet: ',tweet['text'])
              
        created_at = tweet['created_at']
        user_id = tweet['id_str']
        texto=tweet['text']
        
        with open("tweet.json",'a+') as output:
            
            data = {"created_at": created_at,
                    "User_id": user_id,
                    "tweet" : texto }
            
            output.write('{}\n'.format(json.dumps(data)))

except TwitterSearchException as e:
    print(e)
    

df=pd.read_json('tweet.json',lines= True )
df.drop_duplicates(['tweet'],inplace=True)
print(df.head(10))
#%%
#Extração dos jornalistas
import json
from datetime import datetime
from TwitterSearch import *
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import tweepy
import configparser
import time

consumer_key= ''
consumer_secret= ''
access_token=''
access_token_secret=''
df_2 = pd.DataFrame(columns=columns)


#jornalistas=['maurocezar','Wanderley','spimpolo','brunosprado','ranieri_andre','daniel_lian','JovemPanEsporte','UOLEsporte','gazetaesportiva','tresportes','fred.ring','VladirLemos','ArnaldoJRibeiro','williamespn','kfouriandre','pedroivoalmeida','joaoguilherm','gustavohofman','jahumauricio','silvapcaique','vampetaoficial','sampaiojovempan','nilsoncesarjp','flaviopradojpgz','faustofavara','mauro_beting','GiovanniChacon_','opilhado','gaab_diias','PedroMarques21','becharavictoria','ronaldo601','10neto','01velloso','renatabfan','denilsonoficial','UlissesCosta31','JujuSalimeni','EliaJunior','crisdiass','Miltonneves','etironi','piperno','eduribeirotv','missferraz','MCiribelli','onavarromagno','_igorrodrigues','barbaracoelho','andrehernan','Fernanda_Gentil','glendakozlowski','andreolifelipe','AlexEscobar_','ivan_more','flaviocanto','luislacombereal','olucasgutierrez','aovivoexclusivo','xavierjanaina','wcasagrandejr','Caiobaribeiro','SombraTricolor','carloscereto','alexandremotal','marcao97fm','mano97fm','leticiabeppler','domenico97fm','benjaminback','vieira1903','rodrigocapelo','andrealmeidac','ArnaldoJRibeiro','bentocaipira','PortugaFutebol','arenasbt','luizalano','KayLMurray','DanThomasESPN','brunovicari','gianoddi','_paulo_andrade_','pauloamigao','anterogreco','marianaspinelIi','marcelarafael','glausantiago','JuulianaVeiga','LucianoAmaral','fred_b12','desimpedidos','brauneoficial']
           

consumer_key= 'TJupFWV1hEnIC38uh5b9jvA2M'
consumer_secret= '0EOHwgk1kEJqey7yWG44w9VuNuq7QId7OtTv0FzH0RsUT6fEtC'
access_token='1512640269598011392-iWizkYt1iEfrlmrvvZLnPwn3H68Yv4'
access_token_secret='z9a4Y12P8OR41DXvWf6IsDzZaNPGY24DtxommVrHqnSWn'    


#jornalistas=['maurocezar','Wanderley','spimpolo','brunosprado','ranieri_andre','daniel_lian','JovemPanEsporte','UOLEsporte','gazetaesportiva','tresportes','fred.ring','VladirLemos','ArnaldoJRibeiro','williamespn','kfouriandre','pedroivoalmeida','joaoguilherm','gustavohofman','jahumauricio','silvapcaique','vampetaoficial','sampaiojovempan','nilsoncesarjp','flaviopradojpgz','faustofavara','mauro_beting','GiovanniChacon_','opilhado','@gaab_diias','PedroMarques21','becharavictoria','ronaldo601','@10neto','01velloso','renatabfan','denilsonoficial','UlissesCosta31','JujuSalimeni','EliaJunior','crisdiass','Miltonneves','etironi','piperno','eduribeirotv','missferraz','MCiribelli','onavarromagno','_igorrodrigues','barbaracoelho','andrehernan','Fernanda_Gentil','glendakozlowski','@andreolifelipe','AlexEscobar_','ivan_more','flaviocanto','luislacombereal','olucasgutierrez','aovivoexclusivo','xavierjanaina','wcasagrandejr','Caiobaribeiro','@SombraTricolor','carloscereto','alexandremotal','marcao97fm','mano97fm','leticiabeppler','domenico97fm','benjaminback','vieira1903','rodrigocapelo','andrealmeidac','ArnaldoJRibeiro','bentocaipira','PortugaFutebol','arenasbt','luizalano','KayLMurray','DanThomasESPN','brunovicari','gianoddi','_paulo_andrade_','pauloamigao','anterogreco','marianaspinelIi','marcelarafael','glausantiago','JuulianaVeiga','LucianoAmaral','fred_b12','desimpedidos','brauneoficial']
#%%
#Daqui
#jornalistas_total=['maurocezar','Wanderley','spimpolo','brunosprado','ranieri_andre','daniel_lian','JovemPanEsporte','UOLEsporte','gazetaesportiva','tresportes','fred.ring','VladirLemos','ArnaldoJRibeiro','williamespn','kfouriandre','pedroivoalmeida','joaoguilherm','gustavohofman','jahumauricio','silvapcaique','vampetaoficial','sampaiojovempan','nilsoncesarjp','flaviopradojpgz','faustofavara','mauro_beting','GiovanniChacon_','opilhado','gaab_diias','PedroMarques21','becharavictoria','ronaldo601','10neto','01velloso','renatabfan','denilsonoficial','UlissesCosta31','JujuSalimeni','EliaJunior','crisdiass','Miltonneves','etironi','piperno','eduribeirotv','missferraz','MCiribelli','onavarromagno','_igorrodrigues','barbaracoelho','andrehernan','Fernanda_Gentil','glendakozlowski','andreolifelipe','AlexEscobar_','ivan_more','flaviocanto','luislacombereal','olucasgutierrez','aovivoexclusivo','xavierjanaina','wcasagrandejr','Caiobaribeiro','SombraTricolor','carloscereto','alexandremotal','marcao97fm','mano97fm','leticiabeppler','domenico97fm','benjaminback','vieira1903','rodrigocapelo','andrealmeidac','ArnaldoJRibeiro','bentocaipira','PortugaFutebol','arenasbt','luizalano','KayLMurray','DanThomasESPN','brunovicari','gianoddi','_paulo_andrade_','pauloamigao','anterogreco','marianaspinelIi','marcelarafael','glausantiago','JuulianaVeiga','LucianoAmaral','fred_b12','desimpedidos','brauneoficial']
jornalistas=['fred_b12','desimpedidos','brauneoficial','fredringtv','gianoddi']
  
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

conToken = tweepy.API(auth)


columns = ['created_at','User', 'Tweet']

for w in jornalistas:  
   
    start_date = datetime(2022, 4, 16)
    limit=180000
    tweets_iter = tweepy.Cursor(conToken.user_timeline,
                            tweet_mode='extended',
                            screen_name=w,
                            result_type='mixed',
                            until = '2022-04-03',
                            lang='pt', include_entities=True).items(limit)
    
    data = []

    for tweet in tweets_iter:
        data.append([tweet.created_at,tweet.user.screen_name, tweet.full_text])

    df = pd.DataFrame(data, columns=columns)
    nome=w+".csv"

    df.to_csv(nome, sep=';', encoding='utf-8')
    df_2=df_2.append(df)
    print(w)
    time.sleep(5)
    print('próximo scraping')

#

#%%

#Pre processamento
#trabalho com o NLTK


#Trabalho com Stopwords

def RemovendoStopWords(instacia):
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in instacia.split() if not i in stopwords]
    return ("".json(palavras))

stopwords= set(nltk.corpus.stopwords.words('portuguese'))
print(stopwords)

instacia= 'Minerando Dados o maior portal  de Data Science do Brasil'
palavras= [i for i in instacia.split() if not i in stopwords]
print(palavras)


#Trabalho com Stemming

def Stemming(instacia):
    stemmer = nltk.stem.RSLPStemmer()
    palavras=[]
    for w in instacia.split():
        palavras.append(stemmer.stem(w))
    return(" ".join(palavras))

# Limpeza de dados com Re

def Limpeza_dados(instacia):
    instacia=re.sub(r"http\S+","",instacia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    return (instacia)

# Limpeza de dados com B4
def tweet_to_work(tweet):
    tweet = BeautifulSoup(tweet,"htm.parser").get_text()
    tweet = re.sub(r"[a^-zA-Zà-úÀ-Ú0-9]","",tweet.lower())
    tweet = tweet_tokenizer.tokenize(tweet)
    return words

#Lemmatization
#só tem em inglês

wordnet_lemmatizer =  WordNetLemmatizer()

def Lemmatization(instacia):
    palavras = []
    for w in instacia.split():
        palavras.append(wordnet_lemmatizer.Lemmatize(w))
    return (" ".join(palavras))


#Tokenização
tweet_tokenizer = TweetTokenizer()
tweet_tokenizer.tokenize(frase)

tweet_tokenizer= [tweet_tokenizer.tokenize(tweet) for  tweet  in df.tweet]
print(tweet_tokenizer[:5])

def Preprocessing(instacia):
    instacia= re.sub(r"http\S+","",instacia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in instacia.split() if not i in stopwords]
    return (" ".join(palavras))

tweets=[Preprocessing(i) for i  in df.tweet]
print(tweet[:10])

df['Preprocessed']=tweets
print(df.head(15))

#Removendo os caracteres indesejados

def prep_tweets(tweet):
    
    tweet = BeautifulSoup(tweet,"html.parser").get_text()
    tweet = re.sub(r"[^a-zA-Zà-úÀ-Ú0-9]"," ",tweet.lower())
    #words = tweet_tokenizer.tokenize(text)
    return tweet

df['Cleaned_Tweets']= [prep_tweets(tweet) for  tweet in  df['Preprocessed']]
print(df.head(15))


#Contar o N° de palavras
cv=CountVectorizer()
count_matrix= cv.fit_transform(df.Cleaned_Tweets)
print(cv.get_feature_names())
print(count_matrix.toarray())

#criação do df
word_count =pd.DataFrame(cv.get_feature_names(),columns=['Palavras'])
#soma das palavras e conversão  em uma lista
word_count['Frequência']=count_matrix.sum(axis=0).tolist()[0]
word_count= word_count.sort_values('Frequência',ascending=False).reset_index(drop=True)
print(word_count[0:50])#50 palavras mais usadas