import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Recommendation Tool", layout="centered")

# Only show the selection screen first
st.markdown("## 🔄 Select Targeting Recommendation Type to Begin")
selected = st.selectbox(
    "Choose a Mode",
    options=["", "Audience", "Keyword", "Placement"],
    format_func=lambda x: "👉 Select a Mode" if x == "" else f"🔘 {x.upper()} Recommendation"
)

# If no selection yet, stop execution
if selected == "":
    st.stop()

# Continue only if a valid selection is made
mode = selected
text_col = 'ad_group_criterion_keyword_text' if mode == "Keyword" else 'display_name'

# Load Model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# Load Data
@st.cache_data
def load_audience_data():
    return pd.read_csv("your_audience_data.csv")

@st.cache_data
def load_keyword_data():
    return pd.read_csv("your_keyword_data.csv")

@st.cache_data
def load_placement_data():
    return pd.read_csv("your_placement_data.csv")

df = load_audience_data() if mode == "Audience" else load_keyword_data() if mode == "Keyword" else load_placement_data()
df = df.dropna(subset=[text_col, 'country'])

# Show UI now
st.title(f"🔍 {mode} Recommender (Multilingual + Performance Based)")

available_metrics = ['conversions', 'ctr', 'clicks', 'impressions', 'cost_per_conversion', 'conversions_from_interactions_rate']
available_countries = sorted(df['country'].dropna().unique().tolist())

performance_metric = st.selectbox("📊 Choose Performance Metric", available_metrics)
country_filter_list = st.multiselect("🌍 Choose Countries", available_countries, default=available_countries[:3])
user_keyword = st.text_input(f"💡 Enter a {mode} Keyword/Topic", "download")
top_n = st.slider("🔝 Number of Recommendations", 1, 20, 5)

if st.button("Generate Recommendations"):

    def recommend_items(user_keyword, performance_metric, country_filter_list, top_n=5, weight_performance=0.6, weight_similarity=0.4):
        
        if not country_filter_list:
            st.error("❌ Please select at least one country.")
            return pd.DataFrame()
        
        if not country_filter_list or 'ALL' in country_filter_list or 'MULTIPLE' in country_filter_list:
            df_filtered = df.copy()
        else:
            df_filtered = df[df['country'].isin(country_filter_list)]

        df_filtered = df_filtered.dropna(subset=[performance_metric])

        group_cols = [text_col]
        df_grouped = df_filtered.groupby(group_cols, as_index=False).agg({
            'conversions': 'sum',
            'ctr': 'mean',
            'clicks': 'sum',
            'impressions': 'sum',
            'cost_per_conversion': 'mean',
            'conversions_from_interactions_rate': 'mean'
        })
        if not user_keyword.strip():  # If user_keyword is empty or just spaces
            df_grouped = df_grouped.sort_values(by=performance_metric, ascending=False)
            return df_grouped[[text_col, performance_metric]].head(top_n)
        
        user_embedding = model.encode(user_keyword, convert_to_tensor=True)
        text_list = df_grouped[text_col].tolist()
        text_embeddings = model.encode(text_list, convert_to_tensor=True)
        similarities = util.cos_sim(user_embedding, text_embeddings)[0]
        df_grouped['similarity'] = similarities.cpu().numpy()
        df_grouped = df_grouped[df_grouped['similarity'] > 0.5]

        scaler = MinMaxScaler()
        df_grouped[['performance_norm', 'similarity_norm']] = scaler.fit_transform(
            df_grouped[[performance_metric, 'similarity']]
        )

        df_grouped = df_grouped[df_grouped['performance_norm'] > 0.01]

        df_grouped['combined_score'] = (weight_performance * df_grouped['performance_norm'] +
                                        weight_similarity * df_grouped['similarity_norm'])

        df_grouped = df_grouped.sort_values(by='combined_score', ascending=False)
        df_grouped['similarity'] = df_grouped['similarity'].round(4)
        df_grouped[performance_metric] = df_grouped[performance_metric].astype("float64")
        display_cols = [text_col, performance_metric, 'similarity', 'combined_score']
        return df_grouped[display_cols].head(top_n)

    results = recommend_items(user_keyword, performance_metric, country_filter_list, top_n)
    

    st.subheader(f"✅ Top {mode} Suggestions")
    st.dataframe(results)
