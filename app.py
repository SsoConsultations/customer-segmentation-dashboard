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
import io
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Customer Segmentation Dashboard")

# Initialize session state for resetting
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False

st.title("📊 Interactive Customer Segmentation Dashboard")
st.markdown("""
Upload your customer data, select features, choose an algorithm, and discover meaningful customer segments.
**Each step includes easy-to-understand explanations.**
""")

# 1️⃣ Upload Data
st.header("1️⃣ Upload Your Data")

st.markdown("""
Upload a CSV or Excel file containing your customer data. 
Make sure your data has columns representing numeric and categorical information.
""")

uploaded_file = st.file_uploader("Choose a file (.csv or .xlsx)", type=["csv", "xlsx"])

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

if df is not None:
    st.header("2️⃣ Data Overview")
    st.write("Here are the first 5 rows of your data:")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    numeric_cols = [c for c in all_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in all_columns if pd.api.types.is_object_dtype(df[c])]

    st.markdown("Below, select which columns to include in the clustering analysis:")

    # 2️⃣ Feature Selection
    st.subheader("Select Features")

    st.markdown("""
**Numeric Features:**  
These are columns with numbers (e.g., income, age, transaction count).  
They will be scaled to make sure larger numbers don't dominate the analysis.
""")

    selected_numeric = st.multiselect(
        "Select numeric features:",
        numeric_cols,
        default=numeric_cols
    )

    st.markdown("""
**Categorical Features:**  
These are columns with categories (e.g., gender, product type).  
They will be converted into numeric format automatically.
""")

    selected_categorical = st.multiselect(
        "Select categorical features:",
        categorical_cols,
        default=categorical_cols
    )

    st.subheader("Handle Missing Data")

    st.markdown("""
If some rows have missing values, choose how to handle them:
- **Drop Rows:** Remove rows with any missing values in the selected features.
- **Impute:** Fill missing numbers with their column average and categories with the most common value.
""")

    missing_strategy = st.selectbox(
        "Missing Data Handling:",
        ("drop_rows", "impute"),
        format_func=lambda x: x.replace("_", " ").title()
    )

    st.subheader("Train-Test Split")

    st.markdown("""
Split your data into **training** and **testing** sets.  
For example:
- **70% Train:** 70% of rows will be used to create clusters.
- **30% Test:** 30% will be reserved to see how well the clusters generalize.

**Note:** If unsure, set this to 0 to use all data.
""")

    train_ratio = st.slider(
        "Train-Test Split Ratio:",
        min_value=0.0, max_value=0.9, value=0.0, step=0.1
    )

    if not selected_numeric and not selected_categorical:
        st.warning("Please select at least one feature.")
        st.stop()

    # 3️⃣ Preprocessing
    st.header("3️⃣ Preprocessing")

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

    with st.spinner("Preprocessing your data..."):
        scaled_df, df_profile = preprocess_data(df, selected_numeric, selected_categorical, missing_strategy)
    st.success("✅ Preprocessing complete.")
    st.write(scaled_df.head())

    # 4️⃣ Evaluate Clustering
    st.header("4️⃣ Evaluate Clustering Options")

    k_range = range(2, min(11, len(scaled_df)))
    st.markdown("""
We will evaluate **KMeans**, **Gaussian Mixture Model**, and **Agglomerative Clustering** across different numbers of clusters.
""")

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

    scores = evaluate_models(scaled_df, k_range)

    # Plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
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

    # 5️⃣ Recommend Model
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

    best = max(recommendations, key=lambda x: x["silhouette"])
    st.success(f"**Recommended:** {best['algorithm']} with {best['k']} clusters (best Silhouette Score: {best['silhouette']:.3f})")

    # 6️⃣ Choose Final Model
    st.header("6️⃣ Choose Final Model and Run Clustering")

    chosen_algo = st.selectbox("Select Algorithm:", ["KMeans", "Gaussian Mixture Model", "Agglomerative Clustering", "DBSCAN"])

    n_clusters = None
    eps = None
    min_samples = None

    if chosen_algo in ["KMeans", "Gaussian Mixture Model", "Agglomerative Clustering"]:
        n_clusters = st.slider("Number of clusters:", 2, 10, best['k'])
    else:
        eps = st.slider("DBSCAN eps (neighborhood size):", 0.1, 2.0, 0.5, step=0.1)
        min_samples = st.slider("DBSCAN min_samples:", 2, 10, 5)

    if st.button("🚀 Run Clustering"):
        st.session_state.analysis_completed = True
        st.experimental_rerun()

# Reset Button
if st.session_state.analysis_completed:
    st.header("🎉 Analysis Complete!")
    st.markdown("You can now download your results or run a new analysis.")
    if st.button("🔄 Run New Analysis"):
        st.session_state.clear()
        st.experimental_rerun()
