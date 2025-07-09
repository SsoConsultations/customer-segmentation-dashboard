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
from PIL import Image # For loading image

# Suppress warnings for cleaner app output
warnings.filterwarnings("ignore")

# --- Session State Initialization ---
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Page config
st.set_page_config(
    layout="wide",
    page_title="Unsupervised Learning (using Machine Learning)" # Updated title
)

# --- Authentication Logic ---
def login_page():
    # Attempt to load logo for login page
    try:
        logo = Image.open("SsoLogo.jpg")
        st.image(logo, width=150) # Adjust width as needed
    except FileNotFoundError:
        st.warning("SsoLogo.jpg not found. Please ensure it's in the same directory as the script for logo display.")

    st.subheader("Login to Access Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Dummy authentication - replace with secure method in production
        if username == "sso" and password == "sso@123":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Username or Password")

def logout_button():
    if st.session_state.authenticated:
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

# --- Helper Functions ---

# Function to generate cluster summaries
def generate_cluster_summaries(cluster_means_numeric, cluster_cat_proportions, original_df_for_profile, labels, chosen_algo):
    structured_summaries = {}
    total_samples = len(original_df_for_profile)
    
    # Calculate overall means and stds for numeric features
    overall_means = None
    overall_stds = None
    if cluster_means_numeric is not None and not cluster_means_numeric.empty:
        relevant_numeric_cols = [col for col in cluster_means_numeric.columns if col in original_df_for_profile.columns]
        if relevant_numeric_cols:
            overall_means = original_df_for_profile[relevant_numeric_cols].mean()
            overall_stds = original_df_for_profile[relevant_numeric_cols].std()
        
    # Get counts for each cluster from the labels array
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))

    # Sort cluster IDs numerically for consistent report order
    # Ensure index is treated as integers for correct sorting and lookup
    # Filter out NaN/None from cluster_means_numeric.index before converting to int
    valid_cluster_ids = [idx for idx in cluster_means_numeric.index if pd.notna(idx)] if cluster_means_numeric is not None else []
    sorted_cluster_ids = sorted([int(x) for x in valid_cluster_ids])
    
    for cluster_id_int in sorted_cluster_ids:
        # Removed cluster_id_str = str(cluster_id_int) from here to ensure integer access below
        
        cluster_size = label_counts.get(cluster_id_int, 0)
        
        # Special handling for DBSCAN's noise cluster (-1)
        if chosen_algo == "DBSCAN" and cluster_id_int == -1:
             if cluster_size == 0: continue # If no noise points, skip
             structured_summaries[str(cluster_id_int)] = { # Keep string for dict key if that's how it's used elsewhere
                 "cluster_heading": f"Noise Points (DBSCAN): (N={cluster_size}, {(cluster_size / total_samples) * 100:.1f}% of total data)",
                 "numeric_characteristics": [],
                 "categorical_characteristics": [],
                 "persona_implications": "These points could not be assigned to any distinct cluster based on the DBSCAN parameters (eps and min_samples). Potential Implications: *[These might be outliers, or your DBSCAN parameters might need adjustment to capture more clusters.]*"
             }
             continue # Move to next cluster_id

        cluster_percentage = (cluster_size / total_samples) * 100
        
        # FIX 1: Use cluster_id_int for the membership check
        if cluster_id_int not in (cluster_means_numeric.index if cluster_means_numeric is not None else []):
            structured_summaries[str(cluster_id_int)] = { # Keep string for dict key if that's how it's used elsewhere
                "cluster_heading": f"Cluster {cluster_id_int}: (N={cluster_size}, {cluster_percentage:.1f}% of total data)",
                "numeric_characteristics": [],
                "categorical_characteristics": [],
                "persona_implications": "This cluster has 0 members after data processing and therefore no characteristics to display."
            }
            continue # Move to next cluster_id


        numeric_descriptors = []
        if overall_means is not None and not cluster_means_numeric.empty:
            for col in cluster_means_numeric.columns:
                # FIX 2: Use cluster_id_int for .loc access
                cluster_mean = cluster_means_numeric.loc[cluster_id_int, col] 
                
                if col in overall_means and col in overall_stds:
                    overall_mean = overall_means[col]
                    overall_std = overall_stds[col]

                    if overall_std > 0:
                        z_score = (cluster_mean - overall_mean) / overall_std
                        
                        if z_score > 1.2: # Stricter threshold for 'significantly higher'
                            numeric_descriptors.append(f"Higher {col.replace('_', ' ').title()}: Averaging {cluster_mean:.2f} (significantly above overall average of {overall_mean:.2f}).")
                        elif z_score < -1.2: # Stricter threshold for 'significantly lower'
                            numeric_descriptors.append(f"Lower {col.replace('_', ' ').title()}: Averaging {cluster_mean:.2f} (significantly below overall average of {overall_mean:.2f}).")
                    elif cluster_mean > overall_mean and overall_std == 0: # If std is 0, but cluster mean is higher
                        numeric_descriptors.append(f"Higher {col.replace('_', ' ').title()}: Averaging {cluster_mean:.2f}.")
                    elif cluster_mean < overall_mean and overall_std == 0: # If std is 0, but cluster mean is lower
                        numeric_descriptors.append(f"Lower {col.replace('_', ' ').title()}: Averaging {cluster_mean:.2f}.")

        categorical_descriptors = []
        for cat_col, proportions_df in cluster_cat_proportions.items():
            # FIX 3: Use cluster_id_int for membership check
            if cluster_id_int in proportions_df.index: 
                # FIX 4: Use cluster_id_int for .loc access
                cluster_proportions = proportions_df.loc[cluster_id_int].sort_values(ascending=False)
                
                for category, cluster_prop in cluster_proportions.items():
                    if cluster_prop == 0: continue # Skip categories with 0 proportion
                    overall_prop = original_df_for_profile[cat_col].value_counts(normalize=True).get(category, 0)
                    
                    if cluster_prop > 0.3 and cluster_prop > overall_prop * 1.5:
                        categorical_descriptors.append(f"Predominantly {category} (for {cat_col.replace('_', ' ').title()}): {cluster_prop:.1%} of this cluster falls into this category, which is significantly higher than the overall average ({overall_prop:.1%}).")
                    elif cluster_prop > 0.6:
                         categorical_descriptors.append(f"Majorly {category} (for {cat_col.replace('_', ' ').title()}): This category constitutes {cluster_prop:.1%} of the cluster.")
        
        persona_text = "Potential Persona/Implications: *[Consider giving this cluster a descriptive name like 'High-Value Customers' or 'New Engagers' based on its characteristics. Think about what these features mean for your business strategies.]*"

        if not numeric_descriptors and not categorical_descriptors:
            persona_text = "This cluster does not show strong deviations from the overall average in the selected features and may represent an 'average' segment. " + persona_text

        # Keep dictionary key as string for consistency with how it might be used downstream (e.g., in report generation loop)
        structured_summaries[str(cluster_id_int)] = {
            "cluster_heading": f"Cluster {cluster_id_int}: (N={cluster_size}, {cluster_percentage:.1f}% of total data)",
            "numeric_characteristics": numeric_descriptors,
            "categorical_characteristics": categorical_descriptors,
            "persona_implications": persona_text
        }
    return structured_summaries


# Function to create a comprehensive Word report
def create_report(document, algorithm, params, metrics, data_preview_df, pca_plot_bytes, profile_plot_bytes, cluster_means_numeric, cluster_cat_proportions, original_df_for_profile, labels, chosen_algo):
    # Set up document styles
    style = document.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    # Add Logo if exists
    try:
        logo_path = "SsoLogo.jpg"
        if os.path.exists(logo_path):
            document.add_picture(logo_path, width=Inches(1.5))
            last_paragraph = document.paragraphs[-1]
            last_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    except Exception as e:
        st.warning(f"Could not add logo to report: {e}")

    document.add_heading('ML Analysis Report', level=1)
    document.add_paragraph(f"Report generated on: {pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')}")

    # Analysis Overview
    document.add_heading('1. Analysis Overview', level=2)
    document.add_paragraph(
        "This report details the unsupervised learning analysis performed using SSO CONSULTANTS APPLICATION The goal is to group similar data points based on their attributes, enabling targeted strategies."
    )

    # Clustering Parameters
    document.add_heading('2. Clustering Parameters', level=2)
    p = document.add_paragraph()
    p.add_run(f"Algorithm Used: {algorithm.replace('_', ' ').title()}").bold = True
    for param, value in params.items():
        p.add_run(f"\n- {param.replace('_', ' ').title()}: {value}")

    # Model Performance Metrics
    document.add_heading('3. Model Performance Metrics', level=2)
    p = document.add_paragraph()
    p.add_run(f"Silhouette Score: {metrics['silhouette_score']:.4f}\n").bold = True
    p.add_run(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f}\n").bold = True
    p.add_run(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f}").bold = True
    document.add_paragraph(
        "These metrics evaluate the quality of the clusters. Higher Silhouette and Calinski-Harabasz scores indicate better-defined clusters. Lower Davies-Bouldin scores indicate better separation between clusters."
    )

    # Data Preview
    document.add_heading('4. Data Preview (First 5 Rows)', level=2)
    document.add_paragraph("The table below shows the first few rows of the processed dataset, including the assigned cluster.")
    df_html = data_preview_df.head(5).to_html(index=False)
    document.add_paragraph(df_html)

    # Cluster Visualizations
    document.add_heading('5. Cluster Visualizations', level=2)
    document.add_paragraph('PCA Plot: Visualizes clusters in a reduced 2D space.')
    if pca_plot_bytes:
        document.add_picture(io.BytesIO(pca_plot_bytes), width=Inches(6.0))
    else:
        document.add_paragraph("PCA Plot could not be generated (insufficient numeric features or data).")

    document.add_paragraph('Numeric Feature Mean Profiles by Cluster:')
    if profile_plot_bytes:
        document.add_picture(io.BytesIO(profile_plot_bytes), width=Inches(6.0))
    else:
        document.add_paragraph("Numeric Feature Mean Profiles could not be generated (no numeric features).")
    
    # New Section: Cluster Profiles - Tables of Means and Proportions
    document.add_heading('6. Cluster Profiles', level=2)
    
    if cluster_means_numeric is not None and not cluster_means_numeric.empty:
        document.add_paragraph("Average values of numeric features for each cluster:")
        document.add_paragraph(cluster_means_numeric.round(2).to_html())
    else:
        document.add_paragraph("No numeric features selected or processed to display mean profiles.")

    if cluster_cat_proportions:
        document.add_paragraph("Proportions of categorical feature values within each cluster:")
        for col, df_prop in cluster_cat_proportions.items():
            document.add_paragraph(f"  **{col.replace('_', ' ').title()}:**")
            document.add_paragraph((df_prop * 100).round(1).to_html())
    else:
        document.add_paragraph("No categorical features selected or processed to display proportions.")


    # New Section: Cluster Summaries - Formatted as discussed
    document.add_heading('7. Cluster Summaries and Characteristics', level=2)
    document.add_paragraph(
        "Below is a detailed profile for each identified cluster, highlighting their key distinguishing features "
        "compared to the overall dataset. This can help you understand the unique characteristics of each segment "
        "and develop targeted strategies."
    )

    if not (cluster_means_numeric.empty if cluster_means_numeric is not None else True) or cluster_cat_proportions:
        # Pass labels and chosen_algo to generate_cluster_summaries for cluster size calculation and DBSCAN handling
        structured_cluster_summaries = generate_cluster_summaries(cluster_means_numeric, cluster_cat_proportions, original_df_for_profile, labels, chosen_algo)
        
        if structured_cluster_summaries:
            for cluster_id_str, summary_data in structured_cluster_summaries.items():
                p = document.add_paragraph()
                run = p.add_run(f"**{summary_data['cluster_heading']}**")
                run.bold = True
                
                # Check for the "0 members" special message
                if "This cluster has 0 members" in summary_data['persona_implications']:
                    document.add_paragraph(summary_data['persona_implications'])
                else:
                    # Numeric characteristics
                    if summary_data['numeric_characteristics']:
                        for desc in summary_data['numeric_characteristics']:
                            document.add_paragraph(f"- {desc}")
                    
                    # Categorical characteristics
                    if summary_data['categorical_characteristics']:
                        for desc in summary_data['categorical_characteristics']:
                            document.add_paragraph(f"- {desc}")
                    
                    # Persona implications
                    if summary_data['persona_implications']:
                        document.add_paragraph(summary_data['persona_implications'])
        else:
            document.add_paragraph("Unable to generate detailed cluster summaries.")
    else:
        document.add_paragraph("Not enough data (numeric or categorical profiles) to generate detailed cluster summaries.")


    # Add a page break at the end
    document.add_page_break()


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Store original column names for mapping
    st.session_state['original_column_names'] = df.columns.tolist()
    return df

@st.cache_data
def preprocess_data(df, numeric_cols, categorical_cols):
    df_processed = df.copy()
    
    # Impute missing values before encoding/scaling
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    for col in categorical_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    # Scale numeric features
    scaler = StandardScaler()
    if numeric_cols:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Changed sparse to sparse_output
    if categorical_cols:
        encoded_data = encoder.fit_transform(df_processed[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df_processed.index)
        df_processed = pd.concat([df_processed.drop(columns=categorical_cols), encoded_df], axis=1)
    
    return df_processed, scaler, encoder

@st.cache_data
def apply_clustering(df_processed, algorithm_name, n_clusters, eps, min_samples):
    labels = None
    model = None
    
    if algorithm_name == "KMeans":
        # Ensure n_clusters is not greater than the number of samples
        n_clusters = min(n_clusters, len(df_processed))
        if n_clusters <= 1:
            st.warning("KMeans requires n_clusters > 1. Adjusting n_clusters to 2.")
            n_clusters = 2

        try:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init for modern sklearn versions
            labels = model.fit_predict(df_processed)
        except Exception as e:
            st.error(f"KMeans failed: {e}. Please check your data and parameters.")
            return None, None
            
    elif algorithm_name == "DBSCAN":
        try:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(df_processed)
            if len(np.unique(labels)) == 1 and -1 in np.unique(labels): # Only noise cluster
                st.warning("DBSCAN formed only a noise cluster (-1). Try adjusting 'eps' or 'min_samples'.")
        except Exception as e:
            st.error(f"DBSCAN failed: {e}. Please check your data and parameters.")
            return None, None
            
    elif algorithm_name == "Agglomerative Clustering":
        # Ensure n_clusters is not greater than the number of samples
        n_clusters = min(n_clusters, len(df_processed))
        if n_clusters <= 1:
            st.warning("Agglomerative Clustering requires n_clusters > 1. Adjusting n_clusters to 2.")
            n_clusters = 2
        
        try:
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(df_processed)
        except Exception as e:
            st.error(f"Agglomerative Clustering failed: {e}. Please check your data and parameters.")
            return None, None
            
    elif algorithm_name == "Gaussian Mixture Model":
        # Ensure n_clusters is not greater than the number of samples
        n_clusters = min(n_clusters, len(df_processed))
        if n_clusters <= 1:
            st.warning("Gaussian Mixture Model requires n_components > 1. Adjusting n_components to 2.")
            n_clusters = 2
        
        try:
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            model.fit(df_processed)
            labels = model.predict(df_processed)
        except Exception as e:
            st.error(f"Gaussian Mixture Model failed: {e}. Please check your data and parameters.")
            return None, None
            
    return labels, model

@st.cache_data
def calculate_metrics(df_processed, labels):
    metrics = {}
    unique_labels = len(np.unique(labels))
    
    # Filter out noise points for silhouette and Davies-Bouldin scores if DBSCAN
    if -1 in labels:
        # Check if there are any non-noise clusters
        if len(np.unique(labels[labels != -1])) < 2:
            st.warning("Less than 2 non-noise clusters for metrics calculation. Silhouette and Davies-Bouldin scores will not be calculated for DBSCAN if only one or zero clusters are formed (excluding noise).")
            metrics['silhouette_score'] = np.nan
            metrics['davies_bouldin_score'] = np.nan
            metrics['calinski_harabasz_score'] = np.nan # Calinski-Harabasz needs at least 2 clusters also
            return metrics
        
        data_for_metrics = df_processed[labels != -1]
        labels_for_metrics = labels[labels != -1]
    else:
        data_for_metrics = df_processed
        labels_for_metrics = labels

    if len(np.unique(labels_for_metrics)) > 1: # Metrics require at least 2 clusters
        metrics['silhouette_score'] = silhouette_score(data_for_metrics, labels_for_metrics)
        metrics['davies_bouldin_score'] = davies_bouldin_score(data_for_metrics, labels_for_metrics)
    else:
        st.warning("Only one cluster formed (after excluding noise for DBSCAN). Silhouette and Davies-Bouldin scores require at least 2 clusters.")
        metrics['silhouette_score'] = np.nan
        metrics['davies_bouldin_score'] = np.nan
        
    if len(np.unique(labels_for_metrics)) > 1: # Calinski-Harabasz also requires at least 2 clusters
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(data_for_metrics, labels_for_metrics)
    else:
        metrics['calinski_harabasz_score'] = np.nan

    return metrics

@st.cache_data
def generate_plots(df, labels, numeric_cols, pca_components=2):
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels # Keep as int for correct size calculation later

    # PCA Plot
    fig_pca = plt.figure(figsize=(10, 7))
    pca_plot_bytes = None
    
    if len(df_clustered.columns) >= 2 and len(numeric_cols) >= 2 and len(df_clustered) > 1:
        try:
            # Prepare data for PCA - ensure only numeric features are used
            features_for_pca = df_clustered[numeric_cols]
            
            # Drop rows with NaN values before PCA if any (though preprocessing should handle this)
            features_for_pca = features_for_pca.dropna()
            labels_for_pca = df_clustered.loc[features_for_pca.index, 'Cluster'] # Align labels after dropping NaNs

            if len(features_for_pca) > 1 and len(features_for_pca.columns) > 1: # Ensure enough data for PCA
                pca = PCA(n_components=min(pca_components, features_for_pca.shape[1]))
                components = pca.fit_transform(features_for_pca)
                
                # Create a DataFrame for plotting PCA results
                pca_df = pd.DataFrame(data = components, columns = [f'Principal Component {i+1}' for i in range(components.shape[1])])
                pca_df['Cluster'] = labels_for_pca.values # Use .values to avoid index alignment issues if original was filtered

                sns.scatterplot(
                    x=f'Principal Component 1',
                    y=f'Principal Component 2',
                    hue='Cluster',
                    palette='viridis',
                    data=pca_df,
                    legend='full'
                )
                plt.title('PCA of Clusters')
                plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
                plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
                
                # Convert plot to bytes
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                pca_plot_bytes = buf.getvalue()
                plt.close(fig_pca)
            else:
                st.warning("Not enough valid data points or numeric features to perform PCA.")
        except Exception as e:
            st.error(f"Error generating PCA plot: {e}. Check numeric columns and data quality.")
            pca_plot_bytes = None
    else:
        st.warning("PCA plot cannot be generated: Need at least 2 numeric features and more than 1 data point.")
        pca_plot_bytes = None

    # Numeric Feature Mean Profiles (cluster_means_numeric)
    fig_profile_numeric = plt.figure(figsize=(12, 6))
    profile_plot_bytes = None
    cluster_means_numeric = pd.DataFrame() # Initialize as empty DataFrame
    
    if numeric_cols: # Only plot if numeric columns exist
        cluster_means_numeric = df_clustered.groupby('Cluster')[numeric_cols].mean()
        cluster_means_numeric_melted = cluster_means_numeric.reset_index().melt('Cluster', var_name='Feature', value_name='Mean Value')
        
        sns.barplot(x='Feature', y='Mean Value', hue='Cluster', data=cluster_means_numeric_melted, palette='viridis')
        plt.title('Numeric Feature Mean Profiles by Cluster')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        profile_plot_bytes = buf.getvalue()
        plt.close(fig_profile_numeric)
    else:
        profile_plot_bytes = None # Set to None if no numeric columns
    
    # Categorical Feature Proportions (cluster_cat_proportions)
    cluster_cat_proportions = {}
    original_cat_in_df_clustered = [col for col in df_clustered.columns if df_clustered[col].dtype == 'object' and col != 'Cluster']
    
    for col in original_cat_in_df_clustered:
        if col in df_clustered.columns: 
            proportions = df_clustered.groupby('Cluster')[col].value_counts(normalize=True).unstack(fill_value=0)
            cluster_cat_proportions[col] = proportions

    return pca_plot_bytes, profile_plot_bytes, cluster_means_numeric, cluster_cat_proportions

# --- Main App Logic ---
if not st.session_state.authenticated:
    login_page()
else:
    logout_button() # Display logout button if authenticated

    st.title("📊 Unsupervised Learning (using Machine Learning)") # Updated title
    
    st.markdown("""
    Welcome! This app helps you discover customer segments using unsupervised machine learning.
    **Each step comes with simple explanations so you don't need technical knowledge to use it.**
    """)

    # File Upload
    st.header("1️⃣ Upload Your Data")
    st.markdown("""
    Upload your data file in **CSV or Excel** format. Make sure your data has columns with numeric information (like income, age) and/or categories (like gender, region).
    """)

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"], key=st.session_state.file_uploader_key)

    df = None
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.success("File uploaded successfully!")
            st.subheader("Data Preview (First 5 Rows):")
            st.dataframe(df.head())
            st.write(f"Shape of data: {df.shape[0]} rows, {df.shape[1]} columns")

            if df.empty:
                st.warning("Uploaded file is empty. Please upload a file with data.")
                df = None # Reset df to None to prevent further processing
            else:
                # Store original df in session state for report generation
                st.session_state['original_df_for_profile'] = df.copy()

        except Exception as e:
            st.error(f"Error loading file: {e}. Please ensure it's a valid CSV or Excel format.")
            df = None

    if df is not None:
        st.header("2️⃣ Feature Selection")
        st.markdown("""
        Choose the columns you want to use for segmentation. Numeric features will be scaled, and categorical features will be one-hot encoded.
        """)

        all_columns = df.columns.tolist()
        
        # Determine initial suggestions for numeric/categorical
        suggested_numeric = []
        suggested_categorical = []
        for col in all_columns:
            # Heuristic to suggest types
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() > 10 and not col.lower().startswith('id'): # More than 10 unique values, not an ID
                    suggested_numeric.append(col)
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                if df[col].nunique() < 20: # Not too many unique categories
                    suggested_categorical.append(col)

        numeric_cols = st.multiselect(
            "Select Numeric Features for Analysis:",
            all_columns,
            default=suggested_numeric
        )
        categorical_cols = st.multiselect(
            "Select Categorical Features for Analysis:",
            all_columns,
            default=suggested_categorical
        )

        features_for_clustering = numeric_cols + categorical_cols

        if not features_for_clustering:
            st.warning("Please select at least one numeric or categorical feature to proceed.")
        else:
            st.header("3️⃣ Choose Clustering Algorithm & Parameters")
            st.markdown("Select an algorithm and adjust its parameters to find meaningful clusters in your data.")

            col1, col2 = st.columns(2)

            with col1:
                algorithm = st.selectbox(
                    "Select Clustering Algorithm:",
                    ("KMeans", "DBSCAN", "Agglomerative Clustering", "Gaussian Mixture Model")
                )
            
            clustering_params = {}
            labels = None

            with col2:
                if algorithm == "KMeans":
                    n_clusters = st.slider("Number of Clusters (K)", 2, 15, 4)
                    clustering_params["n_clusters"] = n_clusters
                elif algorithm == "DBSCAN":
                    eps = st.slider("Epsilon (eps): Max distance between samples for one to be considered as in the neighborhood of the other.", 0.1, 10.0, 0.5)
                    min_samples = st.slider("Min Samples: Number of samples in a neighborhood for a point to be considered as a core point.", 1, 20, 5)
                    clustering_params["eps"] = eps
                    clustering_params["min_samples"] = min_samples
                elif algorithm == "Agglomerative Clustering":
                    n_clusters = st.slider("Number of Clusters", 2, 15, 4)
                    clustering_params["n_clusters"] = n_clusters
                elif algorithm == "Gaussian Mixture Model":
                    n_clusters = st.slider("Number of Components (Clusters)", 2, 15, 4)
                    clustering_params["n_components"] = n_clusters
            
            st.header("4️⃣ Run Analysis")
            if st.button("🚀 Run Clustering"):
                if not features_for_clustering:
                    st.error("Please select features for clustering in Step 2.")
                else:
                    try:
                        with st.spinner("Processing data and running clustering..."):
                            df_selected = df[features_for_clustering].copy()
                            df_processed, scaler, encoder = preprocess_data(df_selected, numeric_cols, categorical_cols)
                            
                            labels, model = apply_clustering(df_processed, algorithm, **clustering_params)

                        if labels is not None:
                            st.success("Clustering completed successfully!")
                            
                            # Add clusters to original dataframe for display and report
                            df_profile = df.copy()
                            df_profile['Cluster'] = labels

                            unique_clusters = np.unique(labels)
                            if -1 in unique_clusters and algorithm == "DBSCAN":
                                st.info(f"DBSCAN identified {len(unique_clusters) - 1} clusters and 1 noise cluster (-1).")
                            else:
                                st.info(f"Clustering identified {len(unique_clusters)} clusters.")

                            # Calculate metrics only if meaningful clusters are formed
                            if len(np.unique(labels)) > 1: # At least two distinct clusters for metrics
                                metrics = calculate_metrics(df_processed, labels)
                                st.header("5️⃣ Model Performance Metrics")
                                st.markdown("These metrics help evaluate the quality of your clusters:")
                                st.write(f"- **Silhouette Score:** {metrics.get('silhouette_score', np.nan):.4f} (Higher is better, range -1 to 1)")
                                st.write(f"- **Davies-Bouldin Index:** {metrics.get('davies_bouldin_score', np.nan):.4f} (Lower is better)")
                                st.write(f"- **Calinski-Harabasz Index:** {metrics.get('calinski_harabasz_score', np.nan):.4f} (Higher is better)")
                                st.info("Interpretation: Higher Silhouette and Calinski-Harabasz scores indicate better-defined clusters. Lower Davies-Bouldin scores indicate better separation between clusters.")
                            else:
                                st.warning("Not enough distinct clusters formed to calculate full performance metrics.")
                                metrics = {'silhouette_score': np.nan, 'davies_bouldin_score': np.nan, 'calinski_harabasz_score': np.nan}


                            st.header("6️⃣ Cluster Visualizations & Profiles")
                            st.markdown("Explore your clusters visually and understand their characteristics.")

                            pca_plot_bytes, profile_plot_bytes, cluster_means_numeric, cluster_cat_proportions = generate_plots(df_profile, labels, numeric_cols)

                            col_vis1, col_vis2 = st.columns(2)
                            with col_vis1:
                                if pca_plot_bytes:
                                    st.image(pca_plot_bytes, caption="PCA of Clusters", use_column_width=True)
                                else:
                                    st.info("PCA plot not generated (not enough numeric features or data).")
                            with col_vis2:
                                if profile_plot_bytes:
                                    st.image(profile_plot_bytes, caption="Numeric Feature Mean Profiles by Cluster", use_column_width=True)
                                else:
                                    st.info("Numeric feature mean profiles not generated (no numeric features selected).")

                            st.subheader("Numeric Feature Means per Cluster")
                            if not cluster_means_numeric.empty:
                                st.dataframe(cluster_means_numeric.round(2))
                            else:
                                st.info("No numeric features selected to display means.")

                            st.subheader("Categorical Feature Distributions per Cluster")
                            if cluster_cat_proportions:
                                for cat in cluster_cat_proportions:
                                    st.markdown(f"**{cat.replace('_', ' ').title()}:**")
                                    st.dataframe((cluster_cat_proportions[cat] * 100).round(1))
                            else:
                                st.info("No categorical features selected to display distributions.")

                            st.subheader("7️⃣ Download Results")
                            
                            # Prepare data for download (original df + cluster labels)
                            clustered_data_csv = df_profile.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Clustered Data (CSV)",
                                data=clustered_data_csv,
                                file_name=f"clustered_data_{pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )

                            # Generate and download Word Report
                            report_document = Document()
                            create_report(
                                report_document, 
                                algorithm, 
                                clustering_params, 
                                metrics, 
                                df_profile, # Pass df_profile as data_preview_df
                                pca_plot_bytes, 
                                profile_plot_bytes,
                                cluster_means_numeric, # Pass the actual DF
                                cluster_cat_proportions, # Pass the dict
                                st.session_state['original_df_for_profile'], # Pass original df for profiling
                                labels, # Pass labels for cluster size
                                algorithm # Pass chosen_algo for DBSCAN noise handling
                            )
                            
                            report_bytes = io.BytesIO()
                            report_document.save(report_bytes)
                            report_bytes.seek(0) # Rewind to the beginning of the BytesIO object
                            
                            st.download_button(
                                label="📄 Download Comprehensive Report (Word)",
                                data=report_bytes,
                                file_name=f"ML_Analysis_Report_{pd.to_datetime('today').strftime('%Y%m%d_%H%M%S')}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                            st.success("Report generated and ready for download!")


                        else:
                            st.error("Clustering did not form more than one cluster or failed. Please review your data and selected parameters.")
                            st.session_state.analysis_completed = False

                    except Exception as e:
                        st.error(f"An error occurred during clustering or plotting: {e}")
                        st.exception(e) # Display full traceback for debugging
                        st.session_state.analysis_completed = False

    # Reset Buttons and "What's Next" section
    if st.session_state.analysis_completed:
        st.header("🎯 Analysis Complete")
        st.markdown("You can either re-run clustering with different parameters on the current dataset, or clear everything to start fresh with new data.")
        col_reset1, col_reset2 = st.columns(2)
        with col_reset1:
            if st.button("🔄 Rerun Clustering with New Parameters"):
                st.session_state.analysis_completed = False
                st.rerun()
        with col_reset2:
            if st.button("🗑️ Clear All Data & Start Fresh"):
                current_file_uploader_key = st.session_state.get('file_uploader_key', 0)
                st.session_state.clear()
                st.session_state.file_uploader_key = current_file_uploader_key + 1
                st.rerun()
    else:
        if df is None:
            st.info("Please upload a data file (.csv or .xlsx) at the top of the page to begin the analysis.")
