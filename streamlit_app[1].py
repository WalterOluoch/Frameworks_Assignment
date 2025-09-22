import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 CORD-19 Data Explorer")
st.write("Interactive exploration of COVID-19 research papers metadata")
st.markdown("---")

# Load and process data
@st.cache_data
def load_and_process_data():
    try:
        # Load the data
        df = pd.read_csv('metadata.csv')
        
        # Clean the data
        df_clean = df.dropna(subset=['title']).copy()
        df_clean['abstract'] = df_clean['abstract'].fillna('')
        df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
        df_clean['year'] = df_clean['publish_time'].dt.year
        df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))
        df_clean['title_word_count'] = df_clean['title'].apply(lambda x: len(str(x).split()))
        
        return df_clean
    except FileNotFoundError:
        st.error("Error: metadata.csv file not found. Please make sure it's in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data with progress indicator
with st.spinner('Loading and processing data...'):
    df = load_and_process_data()

if df.empty:
    st.error("No data loaded. Please check if metadata.csv exists in the same directory.")
    st.stop()

st.success(f'Data loaded successfully! {len(df):,} records found.')

# Sidebar for filters and info
st.sidebar.header("🔧 Filters & Controls")

# Get available years (excluding NaN)
available_years = sorted(df['year'].dropna().unique())
if available_years:
    min_year = int(min(available_years))
    max_year = int(max(available_years))
else:
    min_year = 2019
    max_year = 2022

# Year range slider
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Journal selection
available_journals = df['journal'].dropna().unique()
if len(available_journals) > 0:
    top_journals = df['journal'].value_counts().head(5).index.tolist()
    journal_filter = st.sidebar.multiselect(
        "Select Journals",
        options=available_journals,
        default=top_journals
    )
else:
    journal_filter = []
    st.sidebar.warning("No journal data available")

# Minimum word count filter
min_words = st.sidebar.slider(
    "Minimum Abstract Word Count",
    min_value=0,
    max_value=500,
    value=0
)

# Filter data based on selections
filtered_df = df[
    (df['year'] >= year_range[0]) & 
    (df['year'] <= year_range[1]) &
    (df['abstract_word_count'] >= min_words)
]

if journal_filter:
    filtered_df = filtered_df[filtered_df['journal'].isin(journal_filter)]

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("⚠️ No data matches your filters. Please adjust your selection.")
    st.stop()

# Display metrics
st.header("📈 Overview Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Papers", f"{len(filtered_df):,}")
col2.metric("Unique Journals", filtered_df['journal'].nunique())
col3.metric("Avg Title Words", f"{filtered_df['title_word_count'].mean():.1f}")
col4.metric("Avg Abstract Words", f"{filtered_df['abstract_word_count'].mean():.1f}")

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Timeline", 
    "🏢 Journals", 
    "📝 Word Analysis", 
    "📊 Statistics", 
    "📋 Data Sample"
])

with tab1:
    st.header("Publication Timeline")
    
    # Publications by year
    year_counts = filtered_df['year'].value_counts().sort_index()
    
    if not year_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        year_counts.plot(kind='bar', ax=ax, color='skyblue', alpha=0.8)
        ax.set_title('Publications by Year', fontsize=16, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Publications', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("No year data available for the selected filters")
    
    # Monthly trend (if enough data)
    if len(filtered_df) > 100 and not filtered_df['publish_time'].isna().all():
        filtered_df['month'] = filtered_df['publish_time'].dt.to_period('M')
        monthly_counts = filtered_df['month'].value_counts().sort_index()
        if not monthly_counts.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly_counts.plot(kind='line', ax=ax, color='red', marker='o', linewidth=2)
            ax.set_title('Monthly Publication Trend', fontsize=16, fontweight='bold')
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Publications', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)

with tab2:
    st.header("Journal Analysis")
    
    # Top journals
    journal_counts = filtered_df['journal'].value_counts().head(15)
    
    if not journal_counts.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        journal_counts.plot(kind='barh', ax=ax, color='lightgreen', alpha=0.8)
        ax.set_title('Top Journals by Publication Count', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Publications', fontsize=12)
        ax.set_ylabel('Journal', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        
        # Journal stats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Journal Statistics")
            st.dataframe(journal_counts.head(10))
        
        with col2:
            st.subheader("Publication Distribution")
            st.write(f"Top 5 journals account for {journal_counts.head(5).sum()/len(filtered_df)*100:.1f}% of publications")
    else:
        st.info("No journal data available for the selected filters")

with tab3:
    st.header("Text Analysis")
    
    # Check if we have titles to analyze
    if not filtered_df['title'].dropna().empty:
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            # Remove common stop words
            stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'on', 'with', 'by', 'as', 'an', 'at'}
            words = [word for word in text.split() if word not in stop_words and len(word) > 2]
            return ' '.join(words)
        
        all_titles = ' '.join(filtered_df['title'].dropna().apply(clean_text))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Word Cloud")
            if all_titles.strip():  # Check if we have any text
                fig, ax = plt.subplots(figsize=(10, 6))
                wordcloud = WordCloud(width=600, height=400, background_color='white').generate(all_titles)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Most Frequent Words in Titles', fontsize=14)
                st.pyplot(fig)
            else:
                st.info("No text available for word cloud")
        
        with col2:
            st.subheader("Top Words")
            if all_titles.strip():
                word_freq = Counter(all_titles.split()).most_common(15)
                if word_freq:
                    words, counts = zip(*word_freq)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(words, counts, color='purple', alpha=0.7)
                    ax.set_title('Top Words in Titles', fontsize=14)
                    ax.set_xlabel('Frequency')
                    ax.grid(axis='x', alpha=0.3)
                    plt.gca().invert_yaxis()
                    st.pyplot(fig)
                else:
                    st.info("No words found for analysis")
            else:
                st.info("No text available for word frequency analysis")
    else:
        st.info("No title data available for text analysis")

with tab4:
    st.header("Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Abstract Length Distribution")
        if not filtered_df['abstract_word_count'].empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            filtered_df['abstract_word_count'].hist(bins=30, ax=ax, color='orange', alpha=0.7)
            ax.set_title('Distribution of Abstract Word Count', fontsize=14)
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("No abstract word count data available")
    
    with col2:
        st.subheader("Title Length Distribution")
        if not filtered_df['title_word_count'].empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            filtered_df['title_word_count'].hist(bins=20, ax=ax, color='lightblue', alpha=0.7)
            ax.set_title('Distribution of Title Word Count', fontsize=14)
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("No title word count data available")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats_df = filtered_df[numeric_cols].describe()
        st.dataframe(stats_df)
    else:
        st.info("No numeric data available for statistics")

with tab5:
    st.header("Data Sample")
    
    st.write(f"Showing {min(10, len(filtered_df))} random papers from the filtered dataset ({len(filtered_df):,} total papers):")
    
    # Sample data display
    sample_cols = [col for col in ['title', 'journal', 'year', 'abstract_word_count', 'publish_time'] if col in filtered_df.columns]
    sample_df = filtered_df[sample_cols].sample(min(10, len(filtered_df)))
    
    # Rename columns for display
    column_names = {
        'title': 'Title',
        'journal': 'Journal',
        'year': 'Year',
        'abstract_word_count': 'Abstract Words',
        'publish_time': 'Publication Date'
    }
    sample_df = sample_df.rename(columns={col: column_names.get(col, col) for col in sample_df.columns})
    
    st.dataframe(sample_df, use_container_width=True)
    
    # Search functionality
    st.subheader("Search Papers")
    search_term = st.text_input("Search in titles:")
    if search_term:
        search_results = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
        st.write(f"Found {len(search_results)} papers matching '{search_term}'")
        if len(search_results) > 0:
            st.dataframe(search_results[['title', 'journal', 'year']].head(10))

# Download functionality
st.sidebar.markdown("---")
st.sidebar.header("💾 Export Data")
if st.sidebar.button("Download Filtered Data as CSV"):
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Click to download",
        data=csv,
        file_name="filtered_cord19_data.csv",
        mime="text/csv"
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **About this app:**
    - Explore COVID-19 research papers from CORD-19 dataset
    - Filter by year, journal, and word count
    - Analyze publication trends and word frequencies
    - Data source: metadata.csv from CORD-19 dataset
    """
)

# Show current filter summary
st.sidebar.markdown("---")
st.sidebar.subheader("Current Filters")
st.sidebar.write(f"**Years:** {year_range[0]} - {year_range[1]}")
st.sidebar.write(f"**Journals:** {len(journal_filter)} selected")
st.sidebar.write(f"**Min Abstract Words:** {min_words}")
st.sidebar.write(f"**Total Papers:** {len(filtered_df):,}")