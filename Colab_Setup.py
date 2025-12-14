# Search Engine for Educational Content - Colab Setup
# ====================================================
# This notebook sets up and runs the NLP project in Google Colab

# %% [markdown]
# # Building a Search Engine for Educational Content
# **CSCI S-89B Final Project - Seymur Hasanov**
# 
# This notebook runs the complete NLP pipeline:
# - LDA Topic Modeling
# - Sentence Transformer Semantic Search  
# - Neural Network Topic Classifier

# %% [markdown]
# ## 1. Setup and Installation

# %%
# Install required packages
!pip install -q gensim>=4.4.0 sentence-transformers tensorflow pyLDAvis wordcloud arxiv nltk

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("✅ Dependencies installed!")

# %% [markdown]
# ## 2. Clone from GitHub

# %%
# Clone your repository
!git clone https://github.com/Seymurhh/Search_engine_educational_project_NLP.git
%cd Search_engine_educational_project_NLP

# %% [markdown]
# ## 3. Import Modules

# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import data_loader
import topic_model
import semantic_search
import neural_classifier

print("✅ Modules imported!")

# %% [markdown]
# ## 4. Load and Preprocess Data

# %%
# Load dataset
print("Loading dataset...")
df = data_loader.load_from_csv("arxiv_dataset.csv")
print(f"✅ Loaded {len(df)} papers")

# Preprocess
print("Preprocessing text...")
processed_docs = [data_loader.preprocess_text(doc) for doc in df['abstract']]
processed_docs = data_loader.make_bigrams(processed_docs)
print(f"✅ Preprocessed {len(processed_docs)} documents")

# %% [markdown]
# ## 5. Topic Modeling with LDA

# %%
# Create dictionary and corpus
print("Creating dictionary and corpus...")
dictionary, corpus = topic_model.create_dictionary_corpus(processed_docs)
print(f"Dictionary size: {len(dictionary)}")

# Train LDA model with 5 topics (optimal based on coherence)
NUM_TOPICS = 5
print(f"\nTraining LDA model with {NUM_TOPICS} topics...")
lda_model = topic_model.train_lda_model(corpus, dictionary, num_topics=NUM_TOPICS)

# Compute coherence score
print("Computing coherence score...")
coherence_score = topic_model.compute_coherence_score(lda_model, processed_docs, dictionary)
print(f"\n🎯 Coherence Score (Cv): {coherence_score:.4f}")

# Display topics
print("\n📊 Discovered Topics:")
print("-" * 60)
topics = topic_model.get_topics(lda_model, num_words=8)
for idx, topic in topics:
    words = [word.split('*')[1].strip().strip('"') for word in topic.split(' + ')]
    print(f"Topic {idx}: {', '.join(words[:6])}")

# %% [markdown]
# ## 6. Topic Distribution Analysis

# %%
from collections import Counter

# Assign topics to documents
topic_counts = Counter()
doc_topics = []

for i, doc_bow in enumerate(corpus):
    topic_dist = lda_model.get_document_topics(doc_bow)
    if topic_dist:
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
        topic_counts[dominant_topic] += 1
        doc_topics.append(dominant_topic)
    else:
        doc_topics.append(-1)

df['dominant_topic'] = doc_topics

print("📈 Topic Distribution:")
print("-" * 40)
for topic_id in range(NUM_TOPICS):
    count = topic_counts.get(topic_id, 0)
    pct = count / len(df) * 100
    print(f"Topic {topic_id}: {count:4d} papers ({pct:5.1f}%)")

# %% [markdown]
# ## 7. Semantic Search with Sentence Transformers

# %%
print("Loading Sentence Transformer model...")
print("(This may take a minute on first run)")

# Initialize semantic search
searcher = semantic_search.SemanticSearch()

# Encode all paper abstracts
paper_embeddings = searcher.encode_papers(tuple(df['abstract'].tolist()))
print(f"✅ Encoded {len(df)} documents!")

# Store embeddings for recommendations
searcher.paper_embeddings = paper_embeddings

# %% 
# Test semantic search
query = "reinforcement learning for robot control"
print(f"\n🔍 Query: '{query}'")
print("-" * 60)

results = searcher.search(query, paper_embeddings, df, top_k=5)
for i, result in enumerate(results):
    title = result['title'][:80]
    score = result['score']
    print(f"{i+1}. [{score:.3f}] {title}...")

# %% [markdown]
# ## 8. Neural Network Topic Classifier

# %%
print("Training Neural Network Classifier...")

# Use the embeddings we already computed (convert from tensor to numpy)
X = paper_embeddings.cpu().numpy()
y = np.array(doc_topics)

# Filter out documents without topics
valid_mask = y >= 0
X = X[valid_mask]
y = y[valid_mask]

print(f"Training on {len(X)} samples with {NUM_TOPICS} classes...")

# Train classifier
classifier, history = neural_classifier.train_classifier(
    X, y, 
    num_topics=NUM_TOPICS,
    epochs=20,
    batch_size=32
)

print(f"\n🎯 Training Accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"🎯 Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")

# %% [markdown]
# ## 9. Visualization

# %%
import matplotlib.pyplot as plt

# Plot topic distribution
fig, ax = plt.subplots(figsize=(10, 5))
topics_list = list(range(NUM_TOPICS))
counts = [topic_counts.get(t, 0) for t in topics_list]

bars = ax.bar(topics_list, counts, color='steelblue', edgecolor='black')
ax.set_xlabel('Topic ID', fontsize=12)
ax.set_ylabel('Number of Papers', fontsize=12)
ax.set_title('Topic Distribution Across Papers', fontsize=14)
ax.set_xticks(topics_list)

# Add value labels
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            str(count), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Training History Plot

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
# 
# | Component | Result |
# |-----------|--------|
# | **Topics Discovered** | 5 |
# | **Coherence Score** | See above |
# | **Search Model** | all-MiniLM-L6-v2 |
# | **Classifier Accuracy** | See above |

# %%
print("\n" + "="*60)
print("✅ PIPELINE COMPLETE!")
print("="*60)
print(f"📊 Topics: {NUM_TOPICS}")
print(f"📈 Coherence: {coherence_score:.4f}")
print(f"🔍 Search: Sentence-BERT indexed {len(df)} documents")
print(f"🧠 Classifier: {history.history['val_accuracy'][-1]:.1%} validation accuracy")
