import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np

def visualize_topics_interactive(lda_model, corpus, dictionary, filepath="lda_visualization.html"):
    """
    Creates an interactive visualization using pyLDAvis e
    """
    print("Generating interactive visualization...")
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, filepath)
    print(f"Visualization saved to {filepath}")
    return vis_data

def create_wordclouds(lda_model, num_topics=5):
    """
    Generates and returns a matplotlib figure for word clouds.
    """
    print("Generating Word Clouds...")
    cols = 3
    rows = (num_topics + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    
    for t in range(num_topics):
        plt.subplot(rows, cols, t + 1)
        # Get top 50 words for the topic
        topic_words = dict(lda_model.show_topic(t, 50))
        wc = WordCloud(background_color="white", max_words=50)
        wc.generate_from_frequencies(topic_words)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic {t}")
    
    plt.tight_layout()
    return plt.gcf()

def plot_topic_trends(df, lda_model, corpus):
    """
    Plots the trend of topics over time.
    Requires 'published' column in df.
    Returns the figure object.
    """
    print("Generating Topic Trend Analysis...")
    
    # 1. Assign dominant topic to each document
    dominant_topics = []
    for bow in corpus:
        # Get topic distribution for the document
        topic_dist = lda_model.get_document_topics(bow)
        # Sort by probability and get the top one
        dominant_topic = sorted(topic_dist, key=lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append(dominant_topic)
    
    df['dominant_topic'] = dominant_topics
    
    # 2. Extract Date information
    df['date'] = pd.to_datetime(df['published'])
    df['year'] = df['date'].dt.year
    df['month_year'] = df['date'].dt.to_period('M').astype(str)

    # Check the time span
    min_year = df['year'].min()
    max_year = df['year'].max()
    
    if max_year - min_year < 2:
        # If data is within 1-2 years, group by Month-Year
        print("Grouping by Month-Year...")
        time_col = 'month_year'
        xlabel = 'Month'
        title = 'Topic Trends Over Time (Monthly)'
    else:
        # Otherwise group by Year
        print("Grouping by Year...")
        time_col = 'year'
        xlabel = 'Year'
        title = 'Topic Trends Over Time (Yearly)'
    
    # 3. Group by Time and Topic
    topic_counts = df.groupby([time_col, 'dominant_topic']).size().reset_index(name='count')
    
    # Sort by time to ensure line plot is correct
    topic_counts = topic_counts.sort_values(time_col)
    
    # 4. Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=topic_counts, x=time_col, y='count', hue='dominant_topic', marker='o', palette='tab10')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Number of Papers')
    plt.legend(title='Topic ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    return plt.gcf()

    plt.grid(True)
    return plt.gcf()

def get_topic_labels(lda_model):
    """
    Returns a dictionary mapping topic IDs to a descriptive label (top 3 words).
    """
    labels = {}
    for t in range(lda_model.num_topics):
        top_words = [w for w, _ in lda_model.show_topic(t, 3)]
        labels[t] = f"Topic {t}: {', '.join(top_words)}"
    return labels

def plot_topic_trends_plotly(df, lda_model, corpus):
    """
    Plots the trend of topics over time using Plotly (Interactive).
    """
    print("Generating Interactive Topic Trend Analysis...")
    
    # 1. Assign dominant topic (if not already done)
    if 'dominant_topic' not in df.columns:
        dominant_topics = []
        for bow in corpus:
            topic_dist = lda_model.get_document_topics(bow)
            dominant_topic = sorted(topic_dist, key=lambda x: x[1], reverse=True)[0][0]
            dominant_topics.append(dominant_topic)
        df['dominant_topic'] = dominant_topics

    # Generate Labels
    topic_labels = get_topic_labels(lda_model)
    df['topic_label'] = df['dominant_topic'].map(topic_labels)

    # 2. Extract Date information
    df['date'] = pd.to_datetime(df['published'])
    df['year'] = df['date'].dt.year
    df['month_year'] = df['date'].dt.to_period('M').astype(str)

    # Check the time span in days
    time_span_days = (df['date'].max() - df['date'].min()).days
    
    if time_span_days < 60:
        # Less than 2 months -> Group by Day
        time_col = 'date_str'
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        title = 'Topic Trends Over Time (Daily)'
    elif time_span_days < 730:
        # Less than 2 years -> Group by Month
        time_col = 'month_year'
        title = 'Topic Trends Over Time (Monthly)'
    else:
        # More than 2 years -> Group by Year
        time_col = 'year'
        title = 'Topic Trends Over Time (Yearly)'
    
    # 3. Group by Time and Topic Label
    topic_counts = df.groupby([time_col, 'topic_label']).size().reset_index(name='count')
    topic_counts = topic_counts.sort_values(time_col)
    
    # 4. Plot with Plotly
    fig = px.scatter(topic_counts, x=time_col, y='count', color='topic_label', 
                     size='count', hover_data=['topic_label', 'count'],
                     title=title, labels={time_col: 'Time', 'count': 'Number of Papers', 'topic_label': 'Topic'})
    
    # Add line to connect points for better trend visibility
    fig.update_traces(mode='lines+markers')
    
    return fig

def get_topic_insights(df, lda_model, corpus):
    """
    Generates a summary dataframe for each topic.
    """
    # Ensure dominant topic is assigned
    if 'dominant_topic' not in df.columns:
        dominant_topics = []
        for bow in corpus:
            topic_dist = lda_model.get_document_topics(bow)
            dominant_topic = sorted(topic_dist, key=lambda x: x[1], reverse=True)[0][0]
            dominant_topics.append(dominant_topic)
        df['dominant_topic'] = dominant_topics

    insights = []
    num_topics = lda_model.num_topics
    
    for t in range(num_topics):
        topic_papers = df[df['dominant_topic'] == t]
        count = len(topic_papers)
        
        
        # Get top words
        top_words = ", ".join([w for w, _ in lda_model.show_topic(t, 7)])
        
        # Get date range and Representative Paper
        if not topic_papers.empty:
            min_date = pd.to_datetime(topic_papers['published']).min().date()
            max_date = pd.to_datetime(topic_papers['published']).max().date()
            date_range = f"{min_date} to {max_date}"
            
            # Find representative paper (highest probability for this topic)
            # We need to look up the original indices. 
            # Since topic_papers is a slice, we can iterate through it to find the max score.
            # However, we don't have the per-document topic score in the DF yet.
            # Let's add it to the DF first in the main loop if possible, or re-calculate here.
            # A simpler way: The df already has 'dominant_topic'. 
            # We can just take the first paper in the sorted list if we had scores.
            # But we don't have scores in DF. Let's just take the most recent one as a proxy or random?
            # Better: Let's calculate the score for these papers.
            
            best_score = -1
            rep_title = "N/A"
            
            # This might be slow if many papers. Optimization: Pre-calculate scores in DF.
            # For now, let's just grab the title of the *first* paper in this group 
            # (which is arbitrary unless sorted).
            # Let's try to find the one with the highest score.
            
            # Iterate over the subset indices
            for idx in topic_papers.index:
                bow = corpus[idx]
                topic_dist = dict(lda_model.get_document_topics(bow))
                score = topic_dist.get(t, 0)
                if score > best_score:
                    best_score = score
                    rep_title = topic_papers.loc[idx, 'title']
            
        else:
            date_range = "N/A"
            rep_title = "N/A"
            
        insights.append({
            'Topic Label': f"Topic {t}: {', '.join([w for w, _ in lda_model.show_topic(t, 3)])}",
            'Top Words': top_words,
            'Representative Paper': rep_title,
            'Paper Count': count,
            'Date Range': date_range
        })
        
    return pd.DataFrame(insights)

def get_topic_growth_metrics(df, lda_model, corpus):
    """
    Calculates growth metrics for each topic to identify emerging trends.
    Returns a DataFrame with growth classification.
    """
    # Ensure dominant topic is assigned
    if 'dominant_topic' not in df.columns:
        dominant_topics = []
        for bow in corpus:
            topic_dist = lda_model.get_document_topics(bow)
            dominant_topic = sorted(topic_dist, key=lambda x: x[1], reverse=True)[0][0]
            dominant_topics.append(dominant_topic)
        df['dominant_topic'] = dominant_topics

    # Extract Date info
    df['date'] = pd.to_datetime(df['published'])
    # Extract Date info
    df['date'] = pd.to_datetime(df['published'])
    
    # Determine granularity based on time span
    time_span_days = (df['date'].max() - df['date'].min()).days
    
    if time_span_days < 60:
        # Daily granularity
        df['time_step'] = ((df['date'] - df['date'].min()) / pd.Timedelta(days=1)).astype(int)
        slope_threshold = 0.1 # Lower threshold for daily slope
    else:
        # Monthly granularity
        df['time_step'] = ((df['date'] - df['date'].min()) / pd.Timedelta(days=30)).astype(int)
        slope_threshold = 0.5

    metrics = []
    topic_labels = get_topic_labels(lda_model)
    
    for t in range(lda_model.num_topics):
        topic_data = df[df['dominant_topic'] == t]
        
        if len(topic_data) < 2:
            trend = "Insufficient Data"
            slope = 0
        else:
            # Group by time step to get counts
            counts = topic_data.groupby('time_step').size().reset_index(name='count')
            
            # If we only have one time point, we can't calculate a slope
            if len(counts) < 2:
                 trend = "Stable"
                 slope = 0
            else:
                # Linear Regression (Polyfit degree 1)
                slope, intercept = np.polyfit(counts['time_step'], counts['count'], 1)
                
                if slope > slope_threshold:
                    trend = "🚀 Emerging"
                elif slope < -slope_threshold:
                    trend = "↘️ Cooling"
                else:
                    trend = "➡️ Stable"

        metrics.append({
            'Topic ID': t,
            'Topic Label': topic_labels[t],
            'Growth Score (Slope)': round(slope, 3),
            'Trend Status': trend,
            'Paper Count': len(topic_data)
        })
        
    return pd.DataFrame(metrics).sort_values('Growth Score (Slope)', ascending=False)

