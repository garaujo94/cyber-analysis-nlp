import streamlit as st
from pathlib import Path
import utils as u



# Load data
all_texts = u.read_all_files(Path('data'))
all_texts['cleaned_text'] = all_texts.text.apply(u.clean_text)
all_texts['cleaned_text_without_stopword'] = all_texts['cleaned_text']\
                                            .apply(lambda x: u.remove_stopwords(x, u.stopwords))

# App Title
st.title('Analyzing Texts')

# Filter Title
st.markdown(f'### Documents to be analyzed')
st.caption('The documents used are quite large, it is likely that it will take a while for the computer to process the changes.')

# Filter Documents
documents_choice = st.multiselect('Documents', all_texts.index.values)
texts_to_work_with = all_texts.loc[documents_choice]

st.markdown('-------')

st.markdown('### Wordcloud')
n_grams_wordcloud_choice = st.multiselect('Select N-grams for Wordcloud', [1, 2, 3])
if n_grams_wordcloud_choice:
    with st.spinner('Please wait...'):
        wordcloud_words = u.get_top_n_grams(all_texts['cleaned_text_without_stopword'],\
                                                    u.stopwords,\
                                                    n_gram=[min(n_grams_wordcloud_choice), \
                                                            max(n_grams_wordcloud_choice)])

        u.plot_wordcloud(wordcloud_words)

st.markdown('-------')

st.markdown('### Frequency Plot')
n_grams_frequency_choice = st.multiselect('Select N-grams for Frequency Plot', [1, 2, 3])
top_n_words = int(st.number_input('Select the amount of words'))
if n_grams_frequency_choice and top_n_words:
    with st.spinner('Loading...'):
        frequency_plot_words = u.get_top_n_grams(all_texts['cleaned_text_without_stopword'],\
                                                    u.stopwords,\
                                                    n_gram=[min(n_grams_frequency_choice), \
                                                            max(n_grams_frequency_choice)])
        u.plot_top_n_grams(frequency_plot_words, top_n_words)
        