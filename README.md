# Building a Search Engine for Educational Content

**CSCI S-89B Introduction to Natural Language Processing - Final Project**

Seymur Hasanov | Harvard Extension School | Fall 2025

## Overview

An intelligent search engine for academic research papers featuring:
- **LDA Topic Modeling** - Discover hidden themes in research papers
- **Semantic Search** - Find papers by meaning, not just keywords
- **Neural Classifier** - Predict paper topics using Keras/TensorFlow

## Quick Start (Google Colab)

1. Open Google Colab
2. Upload the files from this repository
3. Run `Colab_Setup.py` as a notebook

Or clone this repo in Colab:
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
| `Colab_Setup.py` | Ready-to-run Colab notebook |

## Installation (Local)

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
streamlit run app.py
```

## Results

- **Topics**: 5 discovered topics with coherence Cv = 0.4170
- **Search**: Semantic similarity using all-MiniLM-L6-v2
- **Classifier**: ~53% validation accuracy (vs 20% random baseline)
