from bs4 import BeautifulSoup
import re
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer

english_stopwords = set(nltk.corpus.stopwords.words("english"))
def normalize_text(text, remove_html=True, only_words=True, remove_stop_words=True, to_stem=False, stopwords=english_stopwords):
    if remove_html:
        text = BeautifulSoup(text, features="html.parser").get_text()
    if only_words:
        text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    if remove_stop_words:
        text = " ".join([w for w in text.split(" ") if w not in stopwords])
    if to_stem:
        stemmer = PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split(" ")])

    return text.strip()

def text_to_sentences(text, tokenizer):
    return tokenizer.tokenize(text.strip())
    
def save_predict(initial_data, result, filename):
    output = pd.DataFrame(data={"id":initial_data["id"], "sentiment":result})
    output.to_csv(filename, index=False, quoting=3 )