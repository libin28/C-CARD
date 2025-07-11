import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Credit Card Customer Segmentation", layout="wide")

st.title("💳 Credit Card Customer Segmentation Dashboard")
st.markdown("Segment and explore customer behavior using unsupervised learning.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/credit_card-data.csv")
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

df = load_data()

# Display data
if st.checkbox("🔍 Show raw data"):
    st.dataframe(df.head())

# Select numeric features
features = df.select_dtypes(include=np.number)

# Standardize features
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)

# Show cluster distribution
st.subheader("📊 Cluster Distribution")
st.bar_chart(df['Cluster'].value_counts())

# Cluster scatter plot (PCA)
st.subheader("🧭 Visualize Clusters with PCA")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
pca_df = pd.DataFrame(data=pca_result, columns=["PCA1", "PCA2"])
pca_df["Cluster"] = df["Cluster"]
fig2, ax2 = plt.subplots()
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="Set2", data=pca_df, s=80, ax=ax2)
ax2.set_title("PCA Cluster Scatter Plot")
st.pyplot(fig2)


# Feature Comparison
st.subheader("📈 Compare Feature by Cluster")
selected_feature = st.selectbox("Select a feature to compare:", features.columns)

fig, ax = plt.subplots()
sns.boxplot(x='Cluster', y=selected_feature, data=df, palette='Set2', ax=ax)
ax.set_title(f"{selected_feature} by Cluster")
st.pyplot(fig)

# View cluster centers
if st.checkbox("📌 Show Cluster Centers (scaled)"):
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
    st.dataframe(centers_df)

st.markdown("---")