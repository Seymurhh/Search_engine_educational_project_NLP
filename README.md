# Building a Search Engine for Educational Content

## Overview

An intelligent search engine for academic research papers featuring:
- **LDA Topic Modeling** - Discover hidden themes in research papers
- **Semantic Search** - Find papers by meaning, not just keywords
- **Neural Classifier** - Predict paper topics using Keras/TensorFlow

## Quick Start (Google Colab)

Clone this repo in Colab:
```python
!git clone https://github.com/Seymurhh/Search_engine_educational_project_NLP.git
%cd Search_engine_educational_project_NLP
```

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `data_loader.py` | ArXiv API and text preprocessing |
| `topic_model.py` | LDA topic modeling with Gensim |
| `semantic_search.py` | Sentence Transformer search |
| `neural_classifier.py` | Keras neural network classifier |
| `visualize.py` | Visualizations (pyLDAvis, charts) |
| `arxiv_dataset.csv` | Cached dataset (500 papers) |
| `Colab_Setup.py` | colab notebook |

## Installation (Local)

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
streamlit run app.py
```

## Results

- **Topics**: 5 discovered topics with coherence Cv = 0.4170
- **Search**: Semantic similarity using all-MiniLM-L6-v2
- **Classifier**: ~78% validation accuracy (vs 20% random baseline)
- streamlit app allows to explore more topics and more than 2000 papers can be used for topic modeling, semantic search and NN classification tasks
