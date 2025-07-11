import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Streamlit page config
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

# Show raw data
if st.checkbox("🔍 Show raw data"):
    st.dataframe(df.head())

# Select numeric features only
features = df.select_dtypes(include=np.number)

# Standardize the data
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)

# Show cluster distribution
st.subheader("📊 Cluster Distribution")
st.bar_chart(df['Cluster'].value_counts())

# PCA scatter plot of customers
st.subheader("🧭 Visualize Clusters with PCA")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
pca_df = pd.DataFrame(data=pca_result, columns=["PCA1", "PCA2"])
pca_df["Cluster"] = df["Cluster"]

fig2, ax2 = plt.subplots()
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="Set2", data=pca_df, s=80, ax=ax2)
ax2.set_title("PCA Cluster Scatter Plot")
st.pyplot(fig2)

# PCA Line Chart for Cluster Centers
st.subheader("🧩 PCA Line Chart of Cluster Centers")
pca_centers = pca.transform(kmeans.cluster_centers_)
pca_centers_df = pd.DataFrame(pca_centers, columns=["PCA1", "PCA2"])
pca_centers_df["Cluster"] = range(len(pca_centers))

fig4, ax4 = plt.subplots()
sns.lineplot(data=pca_centers_df, x="PCA1", y="PCA2", marker="o", hue="Cluster", palette="Set2", ax=ax4)
ax4.set_title("PCA Line Plot of Cluster Centers")
st.pyplot(fig4)

# Feature comparison boxplot
st.subheader("📈 Compare Feature by Cluster")
selected_feature = st.selectbox("Select a feature to compare:", features.columns)

fig, ax = plt.subplots()
sns.boxplot(x='Cluster', y=selected_feature, data=df, palette='Set2', ax=ax)
ax.set_title(f"{selected_feature} by Cluster")
st.pyplot(fig)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(12, 6))
corr_matrix = features.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax3)
st.pyplot(fig3)

# Show cluster centers (scaled values)
if st.checkbox("📌 Show Cluster Centers (scaled)"):
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
    st.dataframe(centers_df)

st.markdown("---")
st.markdown("🛠 Built with Streamlit | 📊 ML by KMeans | 📌 PCA & Heatmap Visuals")
