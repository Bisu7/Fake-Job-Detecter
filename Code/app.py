import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

file_path = os.path.join(os.path.dirname(__file__), "fake_job_model.pkl")
with open(file_path, 'rb') as f:
    model_data = pickle.load(f)

nb_classifier = model_data['nb_classifier']
clf_log = model_data['clf_log']
clf_num = model_data['clf_num']
count_vectorizer = model_data['count_vectorizer']

# Example dictionary: {location: (fake_count, real_count)}
location_stats = {
    "new york": (30, 120),
    "san francisco": (10, 90),
    "remote": (20, 80),
    "los angeles": (25, 100)
}

def calculate_ratio(location):
    location = location.strip().lower()
    if location in location_stats:
        fake, real = location_stats[location]
        return fake / (fake + real) if (fake + real) > 0 else 0.5
    else:
        return 0.5  # default if location not found

st.title('🕵️ Fake Job Posting Detection App')

tab1, tab2 = st.tabs(["🔍 Predict Fake Job", "📊 EDA"])

# ---------------------------------
# Tab 1: Prediction
# ---------------------------------
with tab1:
    st.header("Enter Job Posting Details")

    location = st.text_input("Job Location (e.g., New York, Remote, etc.)")
    description = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")

    telecommuting = st.selectbox("Telecommuting (Remote Job)", [0, 1])

    if st.button("🚀 Predict"):
        combined_text = f"{description} {requirements}"
        character_count = len(combined_text)
        ratio = calculate_ratio(location)

        text_vector = count_vectorizer.transform([combined_text])
        numerical_input = np.array([[telecommuting, ratio, character_count]])

        pred_log = clf_log.predict(text_vector)
        pred_num = clf_num.predict(numerical_input)
        final_pred = 0 if (pred_log[0] == 0 and pred_num[0] == 0) else 1

        if final_pred == 0:
            st.success("✅ This is likely a **genuine** job posting.")
        else:
            st.error("⚠️ This is likely a **fraudulent** job posting.")

# ---------------------------------
# Tab 2: EDA
# ---------------------------------
with tab2:
    st.header("Upload a Dataset for EDA")

    uploaded_file = st.file_uploader("Upload a cleaned CSV file (like `cleaned_jobs.csv`)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display columns to ensure correct column names
        st.write("📄 Columns of the uploaded file:")
        st.write(df.columns)

        st.write("📄 Preview of Uploaded Data:")
        st.dataframe(df.head())

        st.write("📊 Basic Statistics:")
        st.write(df.describe())

        if 'fraudulent' in df.columns:
            st.subheader("Fraudulent Job Posting Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df, x='fraudulent', ax=ax1)
            st.pyplot(fig1)

        if 'character_count' in df.columns:
            st.subheader("Character Count Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(df['character_count'], kde=True, ax=ax2)
            st.pyplot(fig2)

        if 'ratio' in df.columns:
            st.subheader("Location Ratio Distribution")
            fig3, ax3 = plt.subplots()
            sns.histplot(df['ratio'], kde=True, ax=ax3)
            st.pyplot(fig3)

        # Job Postings Count by Location (with top 10 locations only)
        if 'location' in df.columns:
            st.subheader("Job Postings Count by Location")
            location_counts = df['location'].value_counts().head(10)  # Top 10 locations
            fig4, ax4 = plt.subplots(figsize=(10, 6))  # Adjust size
            sns.barplot(x=location_counts.index, y=location_counts.values, ax=ax4)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")  # Rotate labels
            st.pyplot(fig4)

        # Job Postings Count by Industry (with top 10 industries only)
        if 'industry' in df.columns:
            st.subheader("Job Postings Count by Industry")
            industry_counts = df['industry'].value_counts().head(10)  # Top 10 industries
            fig5, ax5 = plt.subplots(figsize=(10, 6))  # Adjust size
            sns.barplot(x=industry_counts.index, y=industry_counts.values, ax=ax5)
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha="right")  # Rotate labels
            st.pyplot(fig5)

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            st.subheader("Correlation Heatmap")
            corr = numeric_df.corr()
            fig6, ax6 = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax6)
            st.pyplot(fig6)

        # WordCloud Generation (Example)
        if 'description' in df.columns:
            all_descriptions = ' '.join(df['description'].fillna(''))
            wordcloud = WordCloud(width=800, height=400, max_words=100).generate(all_descriptions)
            st.subheader("Word Cloud for Job Descriptions")
            st.image(wordcloud.to_array())
        else:
            st.warning("The dataset does not contain a 'description' column for word cloud generation.")
