import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Import local modules
import data_loader
import topic_model
import visualize
import visualize
from semantic_search import SemanticSearch
import neural_classifier
import numpy as np

# Page Config
st.set_page_config(
    page_title="Search Engine for Educational/Research Topics",
    page_icon="📚",
    layout="wide"
)

# Title and Description
st.title("Search Engine for Educational/Research Topics")
st.markdown("""
This application allows you to explore academic papers from ArXiv using **Topic Modeling (LDA)** and **Semantic Search**.
It demonstrates NLP techniques including:
- **Latent Dirichlet Allocation (LDA)** for discovering hidden topics.
- **Sentence Transformers** for semantic understanding and search.
""")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Data Fetching Config
st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio("Select Source", ["Fetch from ArXiv API", "Load Local Dataset"])

if data_source == "Fetch from ArXiv API":
    query = st.sidebar.text_input("ArXiv Query", value="cat:cs.RO OR cat:cs.AI")
    max_results = st.sidebar.slider("Max Papers", 100, 2000, 500)
else:
    st.sidebar.info("Loading from 'arxiv_dataset.csv'")

# Model Config
st.sidebar.subheader("Topic Modeling")
num_topics = st.sidebar.slider("Number of Topics", 2, 20, 5)

# Load Data
with st.spinner("Processing data..."):
    if data_source == "Fetch from ArXiv API":
        df = data_loader.fetch_arxiv_papers(query=query, max_results=max_results)
        if not df.empty:
            # Save to local corpus
            data_loader.save_to_csv(df, "arxiv_dataset.csv")
            st.toast("Dataset saved to arxiv_dataset.csv")
    else:
        df = data_loader.load_from_csv("arxiv_dataset.csv")
        if df.empty:
            st.error("No local dataset found. Please fetch data from API first.")
            st.stop()
    
    if df.empty:
        st.error("No papers found. Please try a different query.")
        st.stop()
        
    # Preprocess
    processed_docs = [data_loader.preprocess_text(doc) for doc in df['abstract']]
    
    # Bigrams
    processed_docs = data_loader.make_bigrams(processed_docs)
    
    # Initialize Semantic Search
    papers = df.to_dict('records')
    search_engine = SemanticSearch()
    search_engine.paper_embeddings = search_engine.encode_papers(df['abstract'].tolist())

# Main Content Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📄 Data Explorer", "📊 Topic Modeling", "📈 Trend Analysis", "🚀 Research Direction", "🔍 Semantic Search", "🧠 Smart Recommender", "🤖 Neural Classifier"])

with tab1:
    st.header("Fetched Papers")
    st.write(f"Total Papers: {len(df)}")
    st.dataframe(df[['title', 'published', 'categories', 'abstract']])

with tab2:
    st.header("Topic Modeling (LDA)")
    
    # Train Model
    dictionary, corpus = topic_model.create_dictionary_corpus(processed_docs)
    lda_model = topic_model.train_lda_model(corpus, dictionary, num_topics=num_topics)
    
    # Compute Coherence Score
    coherence_score = topic_model.compute_coherence_score(lda_model, processed_docs, dictionary)
    st.metric("Model Coherence Score (C_v)", f"{coherence_score:.4f}", help="Higher is better. Typically 0.4 - 0.7 is good.")
    
    # Show Topics
    st.subheader("Discovered Topics")
    topics = topic_model.get_topics(lda_model)
    for idx, topic in topics:
        st.write(f"**Topic {idx}:** {topic}")
        
    # Interactive Visualization
    st.subheader("Interactive Topic Map")
    try:
        vis_data = visualize.visualize_topics_interactive(lda_model, corpus, dictionary, filepath="lda_visualization.html")
        # Read the saved html file and display it
        with open("lda_visualization.html", 'r') as f:
            html_string = f.read()
        components.html(html_string, width=1300, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Error generating interactive visualization: {e}")

    # Word Clouds
    st.subheader("Topic Word Clouds")
    fig_wc = visualize.create_wordclouds(lda_model, num_topics=num_topics)
    st.pyplot(fig_wc)

    # Topic Insights
    st.subheader("Topic Insights")
    insights_df = visualize.get_topic_insights(df, lda_model, corpus)
    st.dataframe(insights_df)

with tab3:
    st.header("Topic Trends Over Time")
    if 'published' in df.columns:
        # Use Plotly for interactive trend analysis
        fig_trend = visualize.plot_topic_trends_plotly(df, lda_model, corpus)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Published date not available for trend analysis.")

with tab4:
    st.header("🚀 Research Direction Dashboard")
    st.markdown("""
    This dashboard helps identify **Emerging Trends** in the field. 
    It calculates the growth rate of each topic over time to tell you which areas are heating up.
    """)
    
    if 'published' in df.columns:
        growth_df = visualize.get_topic_growth_metrics(df, lda_model, corpus)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        emerging_topics = growth_df[growth_df['Trend Status'].str.contains("Emerging")]
        fastest_growing = growth_df.iloc[0] if not growth_df.empty else None
        
        with col1:
            st.metric("Emerging Topics", len(emerging_topics))
        with col2:
            if fastest_growing is not None:
                st.metric("Fastest Growing", f"Topic {fastest_growing['Topic ID']}")
            else:
                st.metric("Fastest Growing", "N/A")
        with col3:
             st.metric("Total Papers Analyzed", len(df))

        st.subheader("Topic Growth Analysis")
        st.dataframe(growth_df[['Topic Label', 'Trend Status', 'Growth Score (Slope)', 'Paper Count']], use_container_width=True)
        
        st.info("💡 **Tip**: 'Emerging' topics (Positive Slope) represent good opportunities for new research.")
    else:
        st.warning("Published date not available for growth analysis.")

with tab5:
    st.header("Semantic Search")
    st.markdown("Search for papers using natural language. The model understands the *meaning* of your query, not just keywords.")
    
    search_query = st.text_input("Enter search query", "deep learning for autonomous navigation")
    
    if search_query:
        with st.spinner("Searching..."):
            # Initialize Semantic Search
            searcher = SemanticSearch()
            
            # Encode papers (cached)
            paper_embeddings = searcher.encode_papers(df['abstract'].tolist())
            # Perform search
            results = search_engine.search(search_query, paper_embeddings, df, top_k=5)
            
            st.subheader("Top Results")
            for i, res in enumerate(results):
                with st.expander(f"{i+1}. {res['title']} (Score: {res['score']:.4f})"):
                    st.write(f"**Published:** {res['published']}")
                    st.write(f"**Abstract:** {res['abstract']}")
                    st.markdown(f"[Read PDF]({res['pdf_url']})")

with tab6:
    st.header("🧠 Smart Recommender")
    st.markdown("""
    **"If you liked this paper, you might also like..."**
    
    Select a paper from the list below, and our AI will recommend 5 other papers that are semantically similar.
    """)
    
    if not df.empty:
        # Create a list of titles for the dropdown
        paper_titles = [f"{i}: {title}" for i, title in enumerate(df['title'])]
        selected_paper_str = st.selectbox("Select a Paper", paper_titles)
        
        if selected_paper_str:
            # Extract index
            selected_index = int(selected_paper_str.split(':')[0])
            
            st.subheader("Selected Paper")
            st.info(f"**{df.iloc[selected_index]['title']}**\n\n{df.iloc[selected_index]['abstract']}")
            
            if st.button("📝 Generate AI Summary"):
                with st.spinner("Generating summary..."):
                    summary = search_engine.summarize_text(df.iloc[selected_index]['abstract'])
                    st.success(f"**AI Summary**: {summary}")
            
            st.subheader("Recommended Papers")
            recommendations = search_engine.get_recommendations(papers, selected_index, k=5)
            
            for rec in recommendations:
                with st.expander(f"{rec['paper']['title']} (Similarity: {rec['score']:.2f})"):
                    st.markdown(f"**Published**: {rec['paper']['published']}")
                    st.markdown(f"**Abstract**: {rec['paper']['abstract']}")
                    st.markdown(f"[PDF Link]({rec['paper']['pdf_url']})")
    else:
        st.warning("No papers found. Please adjust your search query.")

with tab7:
    st.header("🤖 Neural Topic Classifier (Keras/TensorFlow)")
    st.markdown("""
    This component satisfies the course requirement for a **Neural Network**.
    
    It trains a **Keras Dense Neural Network** to predict the *Topic* of a paper based on its *Semantic Embedding*.
    """)
    
    if 'dominant_topic' in df.columns and search_engine.paper_embeddings is not None:
        
        # Hyperparameters UI
        st.subheader("Hyperparameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.slider("Epochs", 5, 100, 20)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        with col2:
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.3)
        with col3:
            hidden_units = st.slider("Hidden Units (Layer 1)", 32, 512, 128, step=32)
            l2_rate = st.number_input("L2 Regularization", 0.0, 0.1, 0.0, step=0.001, format="%.4f")
        
        if st.button("Train Neural Classifier"):
            with st.spinner("Training Keras Model..."):
                # Prepare Data
                if hasattr(search_engine.paper_embeddings, 'cpu'):
                    embeddings = search_engine.paper_embeddings.cpu().numpy()
                else:
                    embeddings = np.array(search_engine.paper_embeddings)
                labels = df['dominant_topic'].values
                
                # Initialize and Train
                classifier = neural_classifier.TopicClassifier(
                    num_topics=num_topics,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    l2_rate=l2_rate
                )
                loss, accuracy, history = classifier.train(embeddings, labels, epochs=epochs, batch_size=batch_size)
                
                st.success(f"Training Complete! Test Accuracy: **{accuracy:.2%}**")
                
                # Plot Training History
                st.subheader("Training History")
                col1, col2 = st.columns(2)
                
                # Accuracy Plot
                fig_acc, ax_acc = plt.subplots()
                ax_acc.plot(history.history['accuracy'], label='Train')
                ax_acc.plot(history.history['val_accuracy'], label='Validation')
                ax_acc.set_title('Model Accuracy')
                ax_acc.set_xlabel('Epoch')
                ax_acc.set_ylabel('Accuracy')
                ax_acc.legend()
                col1.pyplot(fig_acc)
                
                # Loss Plot
                fig_loss, ax_loss = plt.subplots()
                ax_loss.plot(history.history['loss'], label='Train')
                ax_loss.plot(history.history['val_loss'], label='Validation')
                ax_loss.set_title('Model Loss')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.legend()
                col2.pyplot(fig_loss)
                
                # Save classifier to session state to use for prediction
                st.session_state['classifier'] = classifier
                
        # Prediction Section
        st.markdown("---")
        st.subheader("Test the Classifier")
        user_text = st.text_area("Enter an abstract to classify:", "A new deep learning model for autonomous driving using reinforcement learning.")
        
        if st.button("Classify Text"):
            if 'classifier' in st.session_state:
                # Encode text
                text_embedding = search_engine.model.encode(user_text)
                # Predict
                pred_topic, conf = st.session_state['classifier'].predict(text_embedding)
                
                # Get label
                topic_labels = visualize.get_topic_labels(lda_model)
                label = topic_labels.get(pred_topic, f"Topic {pred_topic}")
                
                st.info(f"**Predicted Topic**: {label}")
                st.write(f"**Confidence**: {conf:.2%}")
            else:
                st.warning("Please train the model first!")
                
    else:
        st.warning("Please run Topic Modeling first (go to the 'Topic Modeling' tab) to generate labels.")

