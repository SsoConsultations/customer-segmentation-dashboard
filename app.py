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
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
import io
import os
import warnings

# Suppress warnings for cleaner app output
warnings.filterwarnings('ignore')

# --- Helper Functions ---

# Function to safely format metrics or return 'N/A'
def format_metric(value):
    return f'{value:.4f}' if isinstance(value, (int, float)) else "N/A"

@st.cache_data
def preprocess_data(df_input, selected_numeric, selected_categorical, missing_strategy):
    """
    Applies selected preprocessing steps to the DataFrame.
    Returns the processed (scaled) DataFrame and the original subset for profiling.
    """
    df_processed = df_input[selected_numeric + selected_categorical].copy()
    initial_rows_count = df_processed.shape[0]

    # Handle Missing Values
    rows_dropped_count = 0
    if missing_strategy == 'drop_rows':
        df_processed.dropna(inplace=True)
        rows_dropped_count = initial_rows_count - df_processed.shape[0]
        if rows_dropped_count > 0:
            st.warning(f"Dropped {rows_dropped_count} rows due to missing values.")
    elif missing_strategy == 'impute_mean':
        for col in selected_numeric:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    elif missing_strategy == 'impute_median':
        for col in selected_numeric:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    elif missing_strategy == 'impute_mode':
        for col in selected_categorical:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    if df_processed.empty:
        st.error("After handling missing values, your dataset became empty. Please adjust your strategy or data.")
        st.stop()

    # Scale Numeric Features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_processed[selected_numeric]), columns=selected_numeric, index=df_processed.index)

    # Encode Categorical Features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = pd.DataFrame(encoder.fit_transform(df_processed[selected_categorical]), columns=encoder.get_feature_names_out(selected_categorical), index=df_processed.index)

    # Combine
    df_final = pd.concat([df_scaled, encoded_features], axis=1)

    return df_final, df_processed # Return original subset for profiling

@st.cache_data
def evaluate_algorithms(data, max_k=10, min_k=2):
    """Evaluates different clustering algorithms for a range of k values."""
    results = {}
    
    # K-Means
    kmeans_scores = {'Silhouette': {}, 'Davies-Bouldin': {}, 'Calinski-Harabasz': {}}
    for k in range(min_k, max_k + 1):
        if len(data) >= k: # Ensure data size is at least k
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(data)
                kmeans_scores['Silhouette'][k] = silhouette_score(data, kmeans_labels)
                kmeans_scores['Davies-Bouldin'][k] = davies_bouldin_score(data, kmeans_labels)
                kmeans_scores['Calinski-Harabasz'][k] = calinski_harabasz_score(data, kmeans_labels)
            except Exception:
                # Handle cases where scores might not be computable (e.g., single cluster)
                kmeans_scores['Silhouette'][k] = np.nan
                kmeans_scores['Davies-Bouldin'][k] = np.nan
                kmeans_scores['Calinski-Harabasz'][k] = np.nan
        else:
            kmeans_scores['Silhouette'][k] = np.nan
            kmeans_scores['Davies-Bouldin'][k] = np.nan
            kmeans_scores['Calinski-Harabasz'][k] = np.nan
    results['KMeans'] = kmeans_scores

    # Gaussian Mixture Model (GMM)
    gmm_scores = {'Silhouette': {}, 'Davies-Bouldin': {}, 'Calinski-Harabasz': {}}
    for k in range(min_k, max_k + 1):
        if len(data) >= k:
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
                gmm_labels = gmm.fit_predict(data)
                gmm_scores['Silhouette'][k] = silhouette_score(data, gmm_labels)
                gmm_scores['Davies-Bouldin'][k] = davies_bouldin_score(data, gmm_labels)
                gmm_scores['Calinski-Harabasz'][k] = calinski_harabasz_score(data, gmm_labels)
            except Exception:
                gmm_scores['Silhouette'][k] = np.nan
                gmm_scores['Davies-Bouldin'][k] = np.nan
                gmm_scores['Calinski-Harabasz'][k] = np.nan
        else:
            gmm_scores['Silhouette'][k] = np.nan
            gmm_scores['Davies-Bouldin'][k] = np.nan
            gmm_scores['Calinski-Harabasz'][k] = np.nan
    results['GMM'] = gmm_scores

    # Agglomerative Clustering
    agglo_scores = {'Silhouette': {}, 'Davies-Bouldin': {}, 'Calinski-Harabasz': {}}
    for k in range(min_k, max_k + 1):
        if len(data) >= k:
            try:
                agglo = AgglomerativeClustering(n_clusters=k)
                agglo_labels = agglo.fit_predict(data)
                agglo_scores['Silhouette'][k] = silhouette_score(data, agglo_labels)
                agglo_scores['Davies-Bouldin'][k] = davies_bouldin_score(data, agglo_labels)
                agglo_scores['Calinski-Harabasz'][k] = calinski_harabasz_score(data, agglo_labels)
            except Exception:
                agglo_scores['Silhouette'][k] = np.nan
                agglo_scores['Davies-Bouldin'][k] = np.nan
                agglo_scores['Calinski-Harabasz'][k] = np.nan
        else:
            agglo_scores['Silhouette'][k] = np.nan
            agglo_scores['Davies-Bouldin'][k] = np.nan
            agglo_scores['Calinski-Harabasz'][k] = np.nan
    results['Agglomerative'] = agglo_scores
    
    return results

@st.cache_data
def train_and_predict(data, algorithm, n_clusters=None, eps=None, min_samples=None):
    """Trains the chosen clustering model and returns labels."""
    model = None
    labels = None
    if algorithm == 'KMeans':
        if n_clusters and len(data) >= n_clusters:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(data)
        else:
            st.error(f"Cannot run KMeans with {n_clusters} clusters. Ensure data size is at least {n_clusters}.")
    elif algorithm == 'GMM':
        if n_clusters and len(data) >= n_clusters:
            model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(data)
        else:
            st.error(f"Cannot run GMM with {n_clusters} components. Ensure data size is at least {n_clusters}.")
    elif algorithm == 'Agglomerative':
        if n_clusters and len(data) >= n_clusters:
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(data)
        else:
            st.error(f"Cannot run Agglomerative Clustering with {n_clusters} clusters. Ensure data size is at least {n_clusters}.")
    elif algorithm == 'DBSCAN':
        if eps is not None and min_samples is not None:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data)
        else:
            st.error("Please provide valid 'eps' and 'min_samples' for DBSCAN.")
    
    if labels is not None and len(np.unique(labels)) > 1: # Ensure more than one cluster is formed
        try:
            silhouette = silhouette_score(data, labels)
            davies_bouldin = davies_bouldin_score(data, labels)
            calinski_harabasz = calinski_harabasz_score(data, labels)
            return labels, model, silhouette, davies_bouldin, calinski_harabasz
        except Exception as e:
            st.warning(f"Could not calculate all metrics for the chosen model (e.g., if only one cluster was formed): {e}")
            return labels, model, np.nan, np.nan, np.nan
    elif labels is not None and len(np.unique(labels)) <= 1:
        st.warning(f"Only {len(np.unique(labels))} cluster(s) formed. Consider adjusting parameters or trying another algorithm.")
        return labels, model, np.nan, np.nan, np.nan
    else:
        return None, None, np.nan, np.nan, np.nan

@st.cache_data
def generate_plots(df, labels, numeric_cols, pca_components=2):
    """Generates PCA plot and cluster profile plots."""
    if labels is None or len(np.unique(labels)) <= 1:
        return None, None
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels.astype(str) # Convert to string for consistent plotting

    # PCA Plot
    pca = PCA(n_components=min(pca_components, len(numeric_cols)))
    components = pca.fit_transform(df_clustered[numeric_cols])
    pca_df = pd.DataFrame(data=components, columns=[f'PC{i+1}' for i in range(components.shape[1])])
    pca_df['Cluster'] = df_clustered['Cluster']

    fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', ax=ax_pca, s=100, alpha=0.7)
    ax_pca.set_title('PCA of Clusters')
    ax_pca.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax_pca.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    ax_pca.legend(title='Cluster')
    plt.close(fig_pca)

    # Cluster Profiles (Mean for Numeric, Proportions for Categorical)
    fig_profile_numeric, ax_profile_numeric = plt.subplots(figsize=(12, 6))
    cluster_means_numeric = df_clustered.groupby('Cluster')[numeric_cols].mean()
    cluster_means_numeric.T.plot(kind='bar', ax=ax_profile_numeric, colormap='viridis')
    ax_profile_numeric.set_title('Numeric Feature Means by Cluster')
    ax_profile_numeric.set_ylabel('Mean Value')
    ax_profile_numeric.tick_params(axis='x', rotation=45)
    ax_profile_numeric.legend(title='Cluster')
    plt.tight_layout()
    plt.close(fig_profile_numeric)

    # Categorical Feature Proportions
    cluster_cat_proportions = {}
    for col in [c for c in df_clustered.columns if df_clustered[c].dtype == 'object' and c != 'Cluster']:
        if col in df_clustered.columns: # Check if column exists after dropping potentially empty ones
            proportions = df_clustered.groupby('Cluster')[col].value_counts(normalize=True).unstack(fill_value=0)
            cluster_cat_proportions[col] = proportions

    return fig_pca, fig_profile_numeric, cluster_means_numeric, cluster_cat_proportions

def create_report(document, algorithm, params, metrics, data_preview_df, pca_plot_bytes, profile_plot_bytes, cluster_means_numeric, cluster_cat_proportions):
    """Generates a comprehensive Word document report."""
    document.add_heading('Customer Segmentation Report', level=1)
    document.add_paragraph(f"Report generated on: {pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')}")

    document.add_heading('1. Analysis Overview', level=2)
    document.add_paragraph(
        "This report details the customer segmentation analysis performed using the Streamlit application. "
        "The goal is to group similar customers based on their attributes, enabling targeted strategies."
    )

    document.add_heading('2. Clustering Parameters', level=2)
    document.add_paragraph(f"Algorithm Used: {algorithm}")
    for param, value in params.items():
        document.add_paragraph(f"- {param.replace('_', ' ').title()}: {value}")
    
    document.add_heading('3. Model Performance Metrics', level=2)
    document.add_paragraph(f"Silhouette Score: {format_metric(metrics['silhouette'])}")
    document.add_paragraph(f"Davies-Bouldin Index: {format_metric(metrics['davies_bouldin'])}")
    document.add_paragraph(f"Calinski-Harabasz Index: {format_metric(metrics['calinski_harabasz'])}")
    document.add_paragraph(
        "These metrics evaluate the quality of the clusters. "
        "Higher Silhouette and Calinski-Harabasz scores indicate better-defined clusters. "
        "Lower Davies-Bouldin scores indicate better separation between clusters."
    )

    document.add_heading('4. Data Preview (First 5 Rows)', level=2)
    table = document.add_table(rows=1, cols=data_preview_df.shape[1])
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(data_preview_df.columns):
        hdr_cells[i].text = col
    for index, row in data_preview_df.iterrows():
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)

    document.add_heading('5. Cluster Visualizations', level=2)
    if pca_plot_bytes:
        document.add_paragraph("PCA Plot: Visualizes clusters in a reduced 2D space.")
        document.add_picture(io.BytesIO(pca_plot_bytes), width=Inches(6))
    else:
        document.add_paragraph("PCA plot could not be generated (e.g., less than 2 numeric features or single cluster).")

    if profile_plot_bytes:
        document.add_paragraph("Numeric Feature Mean Profiles by Cluster:")
        document.add_picture(io.BytesIO(profile_plot_bytes), width=Inches(6))
    else:
        document.add_paragraph("Numeric profile plot could not be generated.")

    document.add_heading('6. Cluster Profiles', level=2)
    
    document.add_paragraph("Average values of numeric features for each cluster:")
    if not cluster_means_numeric.empty:
        table_numeric = document.add_table(rows=1, cols=cluster_means_numeric.shape[1] + 1)
        table_numeric.style = 'Table Grid'
        hdr_cells_numeric = table_numeric.rows[0].cells
        hdr_cells_numeric[0].text = 'Cluster'
        for i, col in enumerate(cluster_means_numeric.columns):
            hdr_cells_numeric[i+1].text = col
        for cluster_id, row_data in cluster_means_numeric.iterrows():
            row_cells = table_numeric.add_row().cells
            row_cells[0].text = str(cluster_id)
            for i, val in enumerate(row_data):
                row_cells[i+1].text = f'{val:.2f}'
    else:
        document.add_paragraph("No numeric cluster means available.")

    document.add_paragraph("\nProportions of categorical feature values within each cluster:")
    if cluster_cat_proportions:
        for cat_col, proportions_df in cluster_cat_proportions.items():
            document.add_paragraph(f"  {cat_col}:")
            table_cat = document.add_table(rows=1, cols=proportions_df.shape[1] + 1)
            table_cat.style = 'Table Grid'
            hdr_cells_cat = table_cat.rows[0].cells
            hdr_cells_cat[0].text = 'Cluster'
            for i, col in enumerate(proportions_df.columns):
                hdr_cells_cat[i+1].text = col
            for cluster_id, row_data in proportions_df.iterrows():
                row_cells = table_cat.add_row().cells
                row_cells[0].text = str(cluster_id)
                for i, val in enumerate(row_data):
                    row_cells[i+1].text = f'{val:.2%}' # Format as percentage
            document.add_paragraph() # Add space between tables
    else:
        document.add_paragraph("No categorical cluster proportions available.")

    document.add_page_break()

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Interactive Customer Segmentation")

st.title("📊 Interactive Customer Segmentation Dashboard")

# Initialize session state for controlling post-execution display
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False

# Function to reset the app
def reset_app():
    st.session_state.clustering_done = False
    st.session_state.data_uploaded = False
    st.cache_data.clear() # Clear all cached data
    st.rerun() # Rerun the app to reset widgets

# --- Upload Section ---
st.header("1. Upload Your Data 📁")
st.info("Upload a CSV or Excel file containing your customer data. The app will analyze this data to find meaningful customer groups.")
uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

df = None
if uploaded_file is not None:
    st.session_state.data_uploaded = True
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        st.write("First 5 rows of your data:")
        st.dataframe(df.head())

        st.subheader("Data Cleaning & Preprocessing Options")
        st.info("Choose how to handle missing values and select the columns you want to use for clustering.")

        # Display missing values
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            st.warning("Missing values detected in the following columns:")
            st.dataframe(missing_data.sort_values(ascending=False))
            missing_strategy = st.selectbox(
                "Select Missing Value Strategy:",
                options=['None', 'drop_rows', 'impute_mean', 'impute_median', 'impute_mode'],
                help="''drop_rows' removes rows with any missing values. 'impute_mean'/'median' fills numeric missing values with the average/middle value. 'impute_mode' fills categorical missing values with the most frequent value."
            )
        else:
            st.info("No missing values found in your dataset.")
            missing_strategy = 'None'

        st.subheader("Select Features for Clustering")
        st.info("Choose the numeric (e.g., Age, Income) and categorical (e.g., Gender, City) columns that describe your customers. These will be used to form the clusters.")

        all_columns = df.columns.tolist()
        selected_numeric_cols = st.multiselect(
            "Select Numeric Features:",
            options=all_columns,
            help="These are columns with numbers that can be scaled (e.g., Age, Income, Spending Score)."
        )
        selected_categorical_cols = st.multiselect(
            "Select Categorical Features:",
            options=all_columns,
            help="These are columns with categories or text labels that will be converted into numbers (e.g., Gender, Region, Education Level)."
        )

        if not selected_numeric_cols and not selected_categorical_cols:
            st.warning("Please select at least one numeric or categorical feature to proceed.")
            st.stop()

        # Check for non-numeric types in selected_numeric_cols
        numeric_errors = []
        for col in selected_numeric_cols:
            # Check if the column is truly numeric or convertible
            if not pd.api.types.is_numeric_dtype(df[col]):
                numeric_errors.append(col)
        if numeric_errors:
            st.error(f"Selected numeric columns contain non-numeric data: {', '.join(numeric_errors)}. Please correct your selection or data.")
            st.stop()
            
        st.subheader("Train-Test Split Ratio")
        st.info("This setting is typically used when you want to evaluate your model on unseen data. For clustering, which is often an exploratory task, we'll use the full dataset for analysis. This option is included for consistency with other ML workflows.")
        
        train_test_split_ratio = st.slider(
            "Select Train-Test Split Ratio (usually 0.0 for clustering exploration):",
            min_value=0.0, max_value=0.5, value=0.0, step=0.05,
            help="For unsupervised learning like clustering, a split of 0.0 is common, meaning the entire dataset is used for analysis."
        )

        st.header("2. Evaluate Clustering Algorithms ✨")
        st.info("Explore how different clustering methods perform with varying numbers of groups. This helps you choose the best approach for your data.")

        min_k_eval = st.slider(
            "Minimum number of clusters (k) to evaluate:",
            min_value=2, max_value=5, value=2,
            help="The smallest number of groups to consider when evaluating algorithms."
        )
        max_k_eval = st.slider(
            "Maximum number of clusters (k) to evaluate:",
            min_value=min_k_eval, max_value=15, value=min(10, min_k_eval + 8),
            help="The largest number of groups to consider. Evaluating too many can be slow."
        )

        if st.button("Evaluate Clustering Algorithms"):
            if not selected_numeric_cols and not selected_categorical_cols:
                st.warning("Please select features first before evaluating algorithms.")
            else:
                with st.spinner("Evaluating algorithms... This may take a moment."):
                    processed_data, _ = preprocess_data(df, selected_numeric_cols, selected_categorical_cols, missing_strategy)
                    if not processed_data.empty:
                        evaluation_results = evaluate_algorithms(processed_data, max_k=max_k_eval, min_k=min_k_eval)

                        st.subheader("Evaluation Plots")
                        st.info("These plots help you visualize how well different algorithms and numbers of clusters perform:")

                        # Plotting Function
                        def plot_scores(scores, title, metric_name, higher_is_better):
                            fig, ax = plt.subplots(figsize=(10, 5))
                            for algo, metrics in scores.items():
                                k_values = sorted(metrics[metric_name].keys())
                                score_values = [metrics[metric_name][k] for k in k_values]
                                ax.plot(k_values, score_values, marker='o', label=algo)
                            ax.set_xlabel('Number of Clusters (k)')
                            ax.set_ylabel(metric_name)
                            ax.set_title(f'{metric_name} for Different Algorithms')
                            ax.legend()
                            ax.grid(True)
                            plt.close(fig)
                            return fig

                        # Silhouette Score Plot
                        st.write("#### Silhouette Score")
                        st.info("Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Higher values (closer to 1) indicate better-defined clusters.")
                        fig_silhouette = plot_scores(evaluation_results, 'Silhouette Score', 'Silhouette', True)
                        st.pyplot(fig_silhouette)

                        # Davies-Bouldin Index Plot
                        st.write("#### Davies-Bouldin Index")
                        st.info("Measures the average similarity ratio of each cluster with its most similar cluster. Lower values (closer to 0) indicate better clustering.")
                        fig_db = plot_scores(evaluation_results, 'Davies-Bouldin Index', 'Davies-Bouldin', False)
                        st.pyplot(fig_db)

                        # Calinski-Harabasz Index Plot
                        st.write("#### Calinski-Harabasz Index")
                        st.info("Also known as the Variance Ratio Criterion. Higher values indicate better-defined clusters.")
                        fig_ch = plot_scores(evaluation_results, 'Calinski-Harabasz Index', 'Calinski-Harabasz', True)
                        st.pyplot(fig_ch)
                        
                        st.subheader("🌟 Model Recommendation 🌟")
                        st.info("Based on standard evaluation practices, here are some suggestions for the best-performing models from your evaluation:")

                        best_silhouette = -np.inf
                        best_db = np.inf
                        best_ch = -np.inf
                        
                        recommended_silhouette = None
                        recommended_db = None
                        recommended_ch = None

                        for algo, metrics in evaluation_results.items():
                            for k, score in metrics['Silhouette'].items():
                                if not np.isnan(score) and score > best_silhouette:
                                    best_silhouette = score
                                    recommended_silhouette = (algo, k, score)
                            
                            for k, score in metrics['Davies-Bouldin'].items():
                                if not np.isnan(score) and score < best_db:
                                    best_db = score
                                    recommended_db = (algo, k, score)
                            
                            for k, score in metrics['Calinski-Harabasz'].items():
                                if not np.isnan(score) and score > best_ch:
                                    best_ch = score
                                    recommended_ch = (algo, k, score)
                        
                        if recommended_silhouette:
                            st.markdown(f"**Best by Silhouette Score:** **{recommended_silhouette[0]}** with **{recommended_silhouette[1]}** clusters (Score: **{format_metric(recommended_silhouette[2])}**).")
                        if recommended_db:
                            st.markdown(f"**Best by Davies-Bouldin Index:** **{recommended_db[0]}** with **{recommended_db[1]}** clusters (Score: **{format_metric(recommended_db[2])}**).")
                        if recommended_ch:
                            st.markdown(f"**Best by Calinski-Harabasz Index:** **{recommended_ch[0]}** with **{recommended_ch[1]}** clusters (Score: **{format_metric(recommended_ch[2])}**).")
                        
                        st.markdown("---") # Separator
                        st.markdown("**Remember:** The final choice should also consider what makes sense for your project's goals and data characteristics.")


                    else:
                        st.error("No data available after preprocessing for evaluation. Check your feature selections and missing value strategy.")

        st.header("3. Final Algorithm & Parameters 🚀")
        st.info("Select the clustering algorithm and its specific settings (like the number of clusters) that you want to apply to your data.")
        
        selected_algorithm = st.selectbox(
            "Choose Clustering Algorithm:",
            options=['KMeans', 'GMM', 'Agglomerative', 'DBSCAN'],
            help="*KMeans*: Simple, fast, assumes spherical clusters. *GMM*: Flexible, assumes clusters are Gaussian distributions. *Agglomerative*: Hierarchical, good for nested clusters. *DBSCAN*: Finds density-based clusters, can find arbitrary shapes, handles noise."
        )

        n_clusters = None
        eps = None
        min_samples = None

        if selected_algorithm in ['KMeans', 'GMM', 'Agglomerative']:
            n_clusters = st.slider(
                "Number of Clusters (k):",
                min_value=2, max_value=max_k_eval, value=3, # Default 3, max based on eval
                help="The desired number of distinct groups to identify in your customer data."
            )
        elif selected_algorithm == 'DBSCAN':
            st.info("DBSCAN finds clusters based on density. You need to define a maximum distance (epsilon) between samples to be considered as in the same neighborhood, and the minimum number of samples required to form a dense region.")
            eps = st.slider(
                "DBSCAN - Epsilon (eps):",
                min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                help="The maximum distance between two samples for one to be considered as in the neighborhood of the other."
            )
            min_samples = st.slider(
                "DBSCAN - Minimum Samples:",
                min_value=1, max_value=20, value=5, step=1,
                help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point (part of a dense region)."
            )

        st.header("4. Run Clustering & Generate Report 📈")
        st.info("Click the button below to run the selected clustering algorithm, view the results, and generate a comprehensive report.")
        
        if st.button("Run Clustering & Generate Report"):
            if not selected_numeric_cols and not selected_categorical_cols:
                st.warning("Please select features before running the clustering.")
            else:
                with st.spinner("Running clustering and generating insights..."):
                    processed_data, df_original_subset = preprocess_data(df, selected_numeric_cols, selected_categorical_cols, missing_strategy)

                    if processed_data.empty:
                        st.error("Processed data is empty. Please check your data, feature selections, and missing value handling.")
                        st.stop()

                    labels, model, silhouette, davies_bouldin, calinski_harabasz = train_and_predict(
                        processed_data,
                        selected_algorithm,
                        n_clusters=n_clusters,
                        eps=eps,
                        min_samples=min_samples
                    )

                    if labels is not None and len(np.unique(labels)) > 1:
                        st.session_state.clustering_done = True # Set flag when clustering is successful
                        st.success(f"Clustering with {selected_algorithm} completed!")

                        st.subheader("Clustering Results & Visualizations")
                        st.write(f"**Selected Algorithm:** {selected_algorithm}")
                        if selected_algorithm != 'DBSCAN':
                            st.write(f"**Number of Clusters (k):** {n_clusters}")
                        else:
                            st.write(f"**Epsilon (eps):** {eps}, **Minimum Samples:** {min_samples}")
                        
                        st.write(f"**Silhouette Score:** {format_metric(silhouette)}")
                        st.write(f"**Davies-Bouldin Index:** {format_metric(davies_bouldin)}")
                        st.write(f"**Calinski-Harabasz Index:** {format_metric(calinski_harabasz)}")

                        fig_pca, fig_profile_numeric, cluster_means_numeric, cluster_cat_proportions = generate_plots(
                            df_original_subset, labels, selected_numeric_cols
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if fig_pca:
                                st.pyplot(fig_pca)
                            else:
                                st.warning("Could not generate PCA plot.")
                        with col2:
                            if fig_profile_numeric:
                                st.pyplot(fig_profile_numeric)
                            else:
                                st.warning("Could not generate numeric cluster profile plot.")

                        st.subheader("Cluster Profiles (Mean Values)")
                        st.info("These tables show the characteristics of each customer group. For numeric features, it's the average value. For categorical features, it's the proportion of each category within that group.")
                        
                        if not cluster_means_numeric.empty:
                            st.dataframe(cluster_means_numeric.round(2))
                        else:
                            st.info("No numeric cluster means to display (check selected features).")

                        if cluster_cat_proportions:
                            st.subheader("Categorical Feature Proportions by Cluster")
                            for col, df_prop in cluster_cat_proportions.items():
                                st.write(f"**{col} Proportions:**")
                                st.dataframe(df_prop.style.format("{:.2%}")) # Format as percentage
                        else:
                            st.info("No categorical cluster proportions to display (check selected features).")


                        # Add Cluster column to original DataFrame subset
                        df_clustered_output = df_original_subset.copy()
                        df_clustered_output['Cluster'] = labels
                        
                        st.header("5. Download Results & Report 📥")
                        st.info("You can download the clustered data or a full report summarizing the analysis.")

                        # Generate Word Report
                        document = Document()
                        pca_plot_bytes = io.BytesIO()
                        if fig_pca:
                            fig_pca.savefig(pca_plot_bytes, format='png', bbox_inches='tight')
                            pca_plot_bytes.seek(0)
                        else:
                            pca_plot_bytes = None

                        profile_plot_bytes = io.BytesIO()
                        if fig_profile_numeric:
                            fig_profile_numeric.savefig(profile_plot_bytes, format='png', bbox_inches='tight')
                            profile_plot_bytes.seek(0)
                        else:
                            profile_plot_bytes = None

                        report_bytes_io = io.BytesIO()
                        create_report(
                            document,
                            selected_algorithm,
                            {'n_clusters': n_clusters, 'eps': eps, 'min_samples': min_samples} if selected_algorithm != 'DBSCAN' else {'eps': eps, 'min_samples': min_samples},
                            {'silhouette': silhouette, 'davies_bouldin': davies_bouldin, 'calinski_harabasz': calinski_harabasz},
                            df_clustered_output.head(5), # Pass data preview
                            pca_plot_bytes.getvalue() if pca_plot_bytes else None,
                            profile_plot_bytes.getvalue() if profile_plot_bytes else None,
                            cluster_means_numeric,
                            cluster_cat_proportions
                        )
                        document.save(report_bytes_io)
                        report_bytes = report_bytes_io.getvalue()
                        report_bytes_io.close()

                        st.download_button(
                            label="Download Comprehensive Report (.docx)",
                            data=report_bytes,
                            file_name=f"Customer_Segmentation_Report_{pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

                        # Download Clustered Data
                        csv_buffer = io.StringIO()
                        df_clustered_output.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download Clustered Data (.csv)",
                            data=csv_buffer.getvalue(),
                            file_name=f"Clustered_Customer_Data_{pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.warning("No valid clusters were formed. Adjust parameters or select different features.")
                else:
                    st.error("Clustering failed. Please check your data and selected parameters.")

if st.session_state.clustering_done:
    st.markdown("---")
    st.header("What's Next? 🤔")
    st.info("You've completed the current analysis. Would you like to download your results or start a new analysis?")
    if st.button("Run New Analysis"):
        reset_app()
else: # Debugging marker: v240709-1
    if not st.session_state.data_uploaded:
        st.info("Please upload a data file (.csv or .xlsx) at the top of the page to begin the analysis.")
