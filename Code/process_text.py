# this code directly copied from https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text


#removing the stopwords
def remove_stopwords(text, tokenizer, stopword_list, is_lower_case=False):
    stop = set(stopwords.words('english'))
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def process_imdb_data():
    # Tokenization of text
    tokenizer = ToktokTokenizer()
    # Setting English stopwords
    stopword_list = nltk.corpus.stopwords.words('english')

    imdb_data = pd.read_csv('./../Datasets/IMDB/IMDB_Dataset.csv')

    # Apply function on review column
    imdb_data['review'] = imdb_data['review'].apply(denoise_text)

    # Apply function on review column
    imdb_data['review'] = imdb_data['review'].apply(remove_special_characters)

    # Apply function on review column
    imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)

    # Apply function on review column
    imdb_data['review'] = imdb_data['review'].apply(remove_stopwords, args=(tokenizer, stopword_list))

    imdb_data.to_csv('./../Datasets/IMDB/Processed_IMDB_Dataset.csv')

    #Tfidf vectorizer
    tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
    #transformed train reviews
    tv_reviews=tv.fit_transform(imdb_data.review)


def transform_imdb_data():
    imdb_data = pd.read_csv('./../Datasets/IMDB/Processed_IMDB_Dataset_Truncated.csv')
    # Count vectorizer
    tv = CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
    # transformed train reviews
    tv_reviews = tv.fit_transform(imdb_data.review)
    print(imdb_data['sentiment'].value_counts())

    # transform sentiment into integer
    sentiment_class = ['negative', 'positive']
    imdb_data['sentiment'] = imdb_data['sentiment'].apply(sentiment_class.index)

    return tv_reviews.toarray(), imdb_data['sentiment'].to_numpy()

def main():
    # process_imdb_data()
    transform_imdb_data()


if __name__ == '__main__':
    main()