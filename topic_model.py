
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import streamlit as st

@st.cache_resource
def create_dictionary_corpus(processed_docs, no_below=20, no_above=0.5):
    """
    Creates a dictionary and corpus from the processed documents.
    """
    dictionary = corpora.Dictionary(processed_docs)
    # Filter out words that occur in less than no_below documents or more than no_above of the documents
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    return dictionary, corpus

@st.cache_resource
def train_lda_model(corpus, _dictionary, num_topics=5, passes=15):
    """
    Trains an LDA model.
    """
    print(f"Training LDA model with {num_topics} topics...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=_dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )
    return lda_model

def compute_coherence_score(lda_model, texts, dictionary):
    """
    Computes the Coherence Score (C_v) for the LDA model.
    """
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda

def get_topics(lda_model, num_words=10):
    """
    Returns the topics and their top words.
    """
    return lda_model.print_topics(num_words=num_words)

