from PyPDF2 import PdfFileReader
import pandas as pd
from pathlib import Path
import os
import re
import string
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st


nlp = spacy.load('en_core_web_lg')
stopwords = nlp.Defaults.stop_words


def read_pdf_file(filepath):
    pdf = open(filepath, "rb")
    reader = PdfFileReader(pdf, strict=False)
    number_of_pages = reader.numPages
    
    pages = []
    
    for i in range(number_of_pages):
        pages.append(reader.getPage(i).extractText())
        
    text = " ".join(pages)
    
    # Check if the document is empty
    if len(re.sub('\s+', ' ', text)) < 10:
        return False, ''
    
    return True, text


def read_all_files(local = Path('..','data')):
    
    files = os.listdir(local)
    year = []
    text = []
    
    for filename in files:
        is_valid, content = read_pdf_file(Path(local, filename))
        if is_valid:
            year.append(filename[:4])
            text.append(content)
    
    texts = pd.DataFrame(text, columns=['text'], index=year)
    
    texts.sort_index(inplace=True)
    
    return texts


def clean_text(text):
    alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", ' ', x)
    punctuation = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
    lower = lambda x: x.lower()
    remove_br = lambda x: re.sub(r""" br """, ' ', x)
    remove_n = lambda x: re.sub(r""" [\r\n]+ """, ' ', x)
    line = lambda x: x.strip()
    linen =  lambda x: re.sub("\n","",x)
    only_letters_and_numbers = lambda x: re.sub('[^a-zA-z0-9\s]','', x)
    double_spaces = lambda x: re.sub('\s+', ' ', x)
    
    text = alphanumeric(text)
    text = punctuation(text)
    text = lower(text)
    text = remove_br(text)
    text = remove_n(text)
    text = line(text)
    text = linen(text)
    text = only_letters_and_numbers(text)
    text = double_spaces(text)
    
    
    return text


def remove_stopwords(text, stopwords):
    text_with_stopwords = [token for token in text.split() if token not in stopwords]
    
    return ' '.join(text_with_stopwords)


def get_top_n_grams(corpus, stopwords, n_gram=[1,1]):
    vec = CountVectorizer(ngram_range=(n_gram[0], n_gram[1]), stop_words=stopwords).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return dict(words_freq)


def plot_top_n_grams(corpus: dict, top_n_grams: int = 30):
    fig, ax = plt.subplots(figsize = (20,16)) # plt.figure(figsize = (16,9))
    
    sns.barplot(x=list(corpus.values())[:top_n_grams],y=list(corpus.keys())[:top_n_grams])
    plt.title(f'Top {top_n_grams} most common ngrams')

    st.pyplot(fig)


def plot_wordcloud(corpus: dict):
    fig, ax = plt.subplots() # plt.figure(figsize = (20,20))
    wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).fit_words(corpus)
    ax.imshow(wc , interpolation = 'bilinear')
    
    st.pyplot(fig)


def search_for_themes(corpus, themes, stopwords) -> dict:
    result = {}
    frequency_by_year = {}
    for year in corpus.index:
        print(f'Searching year: {year}')
        frequency_by_year[year] = get_top_n_grams([corpus.loc[year]], stopwords, n_gram=[1,2])
        result[year] = {}
        for theme in themes:
            result[year][theme] = 0
            result[year]['Total'] = Counter(frequency_by_year[year]).total()
            for words in themes[theme]:
                result[year][theme] += Counter(frequency_by_year[year]).get(words, 0)
    
    return result


def plot_analysis(search_result, plot_type='line', count_type='percentage'):
    '''
    search_result: Output from search_for_themes function
    plot_type: "bar" or "line"
    count_type: "percentage" or "absolute"
    '''
    
    data_to_plot = pd.DataFrame(search_result).T
    
    plt.figure(figsize=(12,8))
    
    if count_type == 'percentage':
        for i, column in enumerate(data_to_plot.drop('Total', axis=1).columns):
            if plot_type == 'line':
                plt.plot(data_to_plot.index, data_to_plot[column]/data_to_plot['Total'], label=column)
            elif plot_type == 'bar':
                plt.bar(data_to_plot.index.to_numpy(dtype=float) + float(i)/5 - 1/5, 
                        data_to_plot[column]/data_to_plot['Total'], 
                        width=0.2,
                        label=column)
            
    elif count_type=='absolute':
        for column in data_to_plot.drop('Total', axis=1).columns:
            plt.plot(data_to_plot.index, data_to_plot[column], label=column)
    plt.grid()
    plt.legend()
    plt.show()
