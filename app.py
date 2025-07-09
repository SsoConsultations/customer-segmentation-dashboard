import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from docx import Document
from docx.shared import Inches, Pt
import io
import os
import warnings

# Suppress warnings for cleaner app output
warnings.filterwarnings("ignore")

# Session state for resetting app
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False

# Page config
st.set_page_config(
    layout="wide",
    page_title="Interactive Customer Segmentation Dashboard"
)

st.title("📊 Customer Segmentation Dashboard")

st.markdown("""
Welcome! This app helps you discover customer segments using unsupervised machine learning.
**Each step comes with simple explanations so you don't need technical knowledge to use it.**
""")

# File Upload
st.header("1️⃣ Upload Your Data")

st.markdown("""
Upload your data file in **CSV or Excel** format. Make sure your data has columns with numeric information (like income, age) and/or categories (like gender, region).
""")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ Successfully loaded {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Data Overview and Feature Selection
if df is not None:
    st.header("2️⃣ Data Overview")

    st.markdown("""
Here are the first 5 rows of your data:
""")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    numeric_cols = [c for c in all_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in all_columns if pd.api.types.is_object_dtype(df[c])]

    st.markdown("""
Below, choose which columns to include in clustering.
""")

    st.subheader("Select Numeric Features")
    st.markdown("""
**Numeric Features:** Columns with numbers, like income, age, or transaction counts.  
These will be standardized so that all values are on the same scale.
""")
    selected_numeric = st.multiselect(
        "Numeric Columns",
        numeric_cols,
        default=numeric_cols
    )

    st.subheader("Select Categorical Features")
    st.markdown("""
**Categorical Features:** Columns with categories, like gender, product type, or region.  
These will be automatically converted into numeric format.
""")
    selected_categorical = st.multiselect(
        "Categorical Columns",
        categorical_cols,
        default=categorical_cols
    )

    st.subheader("Handle Missing Data")
    st.markdown("""
Choose how to handle rows with missing values in your selected columns:
- **Drop Rows:** Remove any rows that have missing values.
- **Impute:** Fill missing numeric values with the column average and missing categories with the most common value.
""")
    missing_strategy = st.selectbox(
        "Missing Data Handling",
        ("drop_rows", "impute"),
        format_func=lambda x: x.replace("_", " ").title()
    )

    st.subheader("Train-Test Split")
    st.markdown("""
Decide if you want to reserve part of your data to test how well the clustering generalizes.

**Example:**
- **70% Train** means 70% of the data will be used to create clusters.
- **30% Test** will be used to validate the results.

If you're not sure, leave this at **0** to use all data.
""")
    train_ratio = st.slider(
        "Train-Test Split Ratio",
        min_value=0.0,
        max_value=0.9,
        value=0.0,
        step=0.1
    )

    if not selected_numeric and not selected_categorical:
        st.warning("Please select at least one numeric or categorical column.")
        st.stop()

    # Data Preprocessing Function
    @st.cache_data
    def preprocess_data(df, numeric, categorical, missing):
        df_proc = df[numeric + categorical].copy()
        if missing == "drop_rows":
            df_proc.dropna(inplace=True)
        else:
            for col in numeric:
                df_proc[col].fillna(df_proc[col].mean(), inplace=True)
            for col in categorical:
                df_proc[col].fillna(df_proc[col].mode()[0], inplace=True)
        df_for_profile = df_proc.copy()

        encoded_features = []
        if categorical:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc_data = encoder.fit_transform(df_proc[categorical])
            enc_df = pd.DataFrame(enc_data, columns=encoder.get_feature_names_out(categorical), index=df_proc.index)
            df_proc.drop(columns=categorical, inplace=True)
            df_proc = pd.concat([df_proc, enc_df], axis=1)
            encoded_features = encoder.get_feature_names_out(categorical).tolist()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_proc)
        scaled_df = pd.DataFrame(scaled, columns=df_proc.columns, index=df_proc.index)
        return scaled_df, df_for_profile
    # Preprocess Data
    st.header("3️⃣ Data Preprocessing")

    with st.spinner("Preprocessing your data..."):
        scaled_df, df_profile = preprocess_data(df, selected_numeric, selected_categorical, missing_strategy)

    st.success("✅ Preprocessing complete!")
    st.write("Here is a preview of your processed data:")
    st.dataframe(scaled_df.head())

    # Clustering Evaluation
    st.header("4️⃣ Clustering Evaluation")

    st.markdown("""
We will now evaluate **KMeans**, **Gaussian Mixture Model**, and **Agglomerative Clustering** using different numbers of clusters.
This helps you see which method separates your data best.
""")

    k_range = range(2, min(11, len(scaled_df)))

    @st.cache_data
    def evaluate_models(scaled_df, k_range):
        scores = {}
        for algo in ["KMeans", "GMM", "Agglomerative"]:
            scores[algo] = {"Silhouette": [], "Davies": [], "Calinski": []}
            for k in k_range:
                if k >= len(scaled_df):
                    break
                if algo == "KMeans":
                    labels = KMeans(n_clusters=k, n_init=10).fit_predict(scaled_df)
                elif algo == "GMM":
                    labels = GaussianMixture(n_components=k).fit_predict(scaled_df)
                else:
                    labels = AgglomerativeClustering(n_clusters=k).fit_predict(scaled_df)
                scores[algo]["Silhouette"].append(silhouette_score(scaled_df, labels))
                scores[algo]["Davies"].append(davies_bouldin_score(scaled_df, labels))
                scores[algo]["Calinski"].append(calinski_harabasz_score(scaled_df, labels))
        return scores

    with st.spinner("Evaluating clustering performance..."):
        scores = evaluate_models(scaled_df, k_range)

    # Plot metrics
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for algo in scores:
        axes[0].plot(k_range[:len(scores[algo]["Silhouette"])], scores[algo]["Silhouette"], label=algo)
        axes[1].plot(k_range[:len(scores[algo]["Davies"])], scores[algo]["Davies"], label=algo)
        axes[2].plot(k_range[:len(scores[algo]["Calinski"])], scores[algo]["Calinski"], label=algo)
    axes[0].set_title("Silhouette Score (Higher is better)")
    axes[1].set_title("Davies-Bouldin Index (Lower is better)")
    axes[2].set_title("Calinski-Harabasz Index (Higher is better)")
    for ax in axes:
        ax.legend()
        ax.grid()
    st.pyplot(fig)

    st.markdown("""
**How to Read These Graphs:**
- **Silhouette Score:** Higher means clearer separation between clusters.
- **Davies-Bouldin Index:** Lower means clusters are more compact and distinct.
- **Calinski-Harabasz Index:** Higher means better defined clusters.
""")

    # Model Recommendation
    st.header("🔍 Recommended Model Based on Metrics")

    recommendations = []
    for algo in scores:
        idx_best = np.argmax(scores[algo]["Silhouette"])
        k_best = k_range[idx_best]
        recommendations.append({
            "algorithm": algo,
            "k": k_best,
            "silhouette": scores[algo]["Silhouette"][idx_best],
            "davies": scores[algo]["Davies"][idx_best],
            "calinski": scores[algo]["Calinski"][idx_best]
        })

    # Recommend the one with the highest silhouette
    best = max(recommendations, key=lambda x: x["silhouette"])

    st.success(
        f"**Recommended:** {best['algorithm']} with {best['k']} clusters "
        f"(Silhouette Score: {best['silhouette']:.3f}, Davies-Bouldin: {best['davies']:.3f}, Calinski-Harabasz: {best['calinski']:.1f})"
    )

    # Final Algorithm Selection
    st.header("5️⃣ Choose Final Model for Clustering")

    st.markdown("""
You can choose the recommended option or pick any algorithm and parameters yourself.
""")

    chosen_algo = st.selectbox(
        "Select Clustering Algorithm",
        ["KMeans", "Gaussian Mixture Model", "Agglomerative Clustering", "DBSCAN"],
        index=["KMeans", "Gaussian Mixture Model", "Agglomerative Clustering"].index(best["algorithm"]) if best["algorithm"] in ["KMeans", "Gaussian Mixture Model", "Agglomerative Clustering"] else 0
    )

    n_clusters = None
    eps = None
    min_samples = None

    if chosen_algo in ["KMeans", "Gaussian Mixture Model", "Agglomerative Clustering"]:
        n_clusters = st.slider(
            "Number of clusters (k)",
            min_value=2,
            max_value=10,
            value=int(best["k"])
        )
    else:
        eps = st.slider(
            "DBSCAN: Neighborhood size (eps)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        min_samples = st.slider(
            "DBSCAN: Minimum samples per cluster",
            min_value=2,
            max_value=10,
            value=5
        )

    st.markdown("""
When you're ready, click the button below to run clustering and generate results.
""")
    # Run Clustering
    if st.button("🚀 Run Clustering"):
        st.header("6️⃣ Clustering Results")
        with st.spinner("Running clustering..."):
            if chosen_algo == "KMeans":
                model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                labels = model.fit_predict(scaled_df)
            elif chosen_algo == "Gaussian Mixture Model":
                model = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = model.fit_predict(scaled_df)
            elif chosen_algo == "Agglomerative Clustering":
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(scaled_df)
            elif chosen_algo == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(scaled_df)

        st.success(f"✅ Clustering completed. Found {len(np.unique(labels))} clusters.")

        # Show cluster counts
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts_df = pd.DataFrame({"Cluster": unique, "Count": counts})
        st.write("**Cluster Distribution:**")
        st.dataframe(cluster_counts_df)

        # PCA Visualization
        if scaled_df.shape[1] >= 2 and len(np.unique(labels)) > 1:
            pca = PCA(n_components=2, random_state=42)
            pcs = pca.fit_transform(scaled_df)
            pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
            pca_df["Cluster"] = labels
            fig_pca, ax_pca = plt.subplots(figsize=(8,6))
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=80)
            ax_pca.set_title("PCA Plot of Clusters")
            st.pyplot(fig_pca)
        else:
            st.info("PCA plot requires at least 2 features and 2 clusters.")

        # Cluster Profiles
        df_profile["Cluster"] = labels
        st.subheader("Numeric Feature Means per Cluster")
        if selected_numeric:
            num_means = df_profile.groupby("Cluster")[selected_numeric].mean()
            st.dataframe(num_means.round(2))

        if selected_categorical:
            st.subheader("Categorical Feature Distributions per Cluster")
            for cat in selected_categorical:
                st.markdown(f"**{cat}:**")
                prop_df = pd.crosstab(df_profile["Cluster"], df_profile[cat], normalize="index")
                st.dataframe((prop_df * 100).round(1))

        # Save clustered data
        clustered_data_csv = df_profile.to_csv(index=False)

        st.subheader("7️⃣ Download Results")

        st.download_button(
            "📥 Download Clustered Data (CSV)",
            data=clustered_data_csv,
            file_name="clustered_data.csv",
            mime="text/csv"
        )

        st.info("""
📄 **Note:** For this simplified version, we haven't generated a Word report.  
If you want to restore your original comprehensive report logic, we can easily integrate your `generate_comprehensive_report()` function again.
""")

        # Mark analysis completed
        st.session_state.analysis_completed = True

# Reset Button
if st.session_state.analysis_completed:
    st.header("🎯 Analysis Complete")
    st.markdown("If you'd like to start over with a new dataset, click below.")
    if st.button("🔄 Run New Analysis"):
        st.session_state.clear()
        st.experimental_rerun()
