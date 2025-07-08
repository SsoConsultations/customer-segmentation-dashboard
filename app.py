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
    elif missing_strategy == 'impute':
        for col in selected_numeric:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
        for col in selected_categorical:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

    # Store the df_processed (original features, after missing handling) for profiling later
    df_for_profiling = df_processed.copy()

    # Categorical Encoding (One-Hot Encoding)
    encoded_features = []
    if selected_categorical:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = encoder.fit_transform(df_processed[selected_categorical])
        encoded_feature_names = encoder.get_feature_names_out(selected_categorical)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df_processed.index)
        df_processed = df_processed.drop(columns=selected_categorical)
        df_processed = pd.concat([df_processed, encoded_df], axis=1)
        encoded_features = encoded_feature_names.tolist()

    all_features_for_scaling = selected_numeric + encoded_features

    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_processed[all_features_for_scaling])
    scaled_df = pd.DataFrame(scaled_data, columns=all_features_for_scaling, index=df_processed.index)

    return scaled_df, df_for_profiling, rows_dropped_count, all_features_for_scaling

@st.cache_data
def evaluate_algorithms(scaled_df_input, k_range_eval):
    """
    Evaluates KMeans, GMM, Agglomerative Clustering for a range of k values.
    Returns scores for plotting.
    """
    kmeans_scores = {'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []}
    gmm_scores = {'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []}
    agglo_scores = {'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []}

    if scaled_df_input.empty:
        return kmeans_scores, gmm_scores, agglo_scores # Return empty if no data

    for k in k_range_eval:
        if k >= len(scaled_df_input): # Avoid error if k is too large for dataset size
            for score_dict in [kmeans_scores, gmm_scores, agglo_scores]:
                for key in score_dict:
                    score_dict[key].extend([np.nan] * (max(k_range_eval) - k + 1))
            break

        # KMeans
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans_model.fit_predict(scaled_df_input)
        kmeans_scores['Silhouette'].append(silhouette_score(scaled_df_input, kmeans_labels))
        kmeans_scores['Davies-Bouldin'].append(davies_bouldin_score(scaled_df_input, kmeans_labels))
        kmeans_scores['Calinski-Harabasz'].append(calinski_harabasz_score(scaled_df_input, kmeans_labels))

        # Gaussian Mixture Models (GMM)
        gmm_model = GaussianMixture(n_components=k, random_state=42)
        gmm_labels = gmm_model.fit_predict(scaled_df_input)
        gmm_scores['Silhouette'].append(silhouette_score(scaled_df_input, gmm_labels))
        gmm_scores['Davies-Bouldin'].append(davies_bouldin_score(scaled_df_input, gmm_labels))
        gmm_scores['Calinski-Harabasz'].append(calinski_harabasz_score(scaled_df_input, gmm_labels))

        # Agglomerative Clustering
        agglo_model = AgglomerativeClustering(n_clusters=k)
        agglo_labels = agglo_model.fit_predict(scaled_df_input)
        agglo_scores['Silhouette'].append(silhouette_score(scaled_df_input, agglo_labels))
        agglo_scores['Davies-Bouldin'].append(davies_bouldin_score(scaled_df_input, agglo_labels))
        agglo_scores['Calinski-Harabasz'].append(calinski_harabasz_score(scaled_df_input, agglo_labels))

    return kmeans_scores, gmm_scores, agglo_scores

@st.cache_resource # Use cache_resource for models
def train_clustering_model(scaled_df_input, algorithm, n_clusters, eps, min_samples):
    """
    Trains the chosen clustering model.
    Returns the trained model and cluster labels.
    """
    model = None
    cluster_labels = None
    n_noise_points = 0

    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(scaled_df_input)
    elif algorithm == "Gaussian Mixture Model":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        model.fit(scaled_df_input)
        cluster_labels = model.predict(scaled_df_input)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(scaled_df_input)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = model.fit_predict(scaled_df_input)
        n_noise_points = list(cluster_labels).count(-1)

    return model, cluster_labels, n_noise_points

@st.cache_data
def generate_cluster_profiles(df_original_subset, cluster_labels, selected_numeric, selected_categorical):
    """
    Generates numeric means and categorical proportions for each cluster.
    """
    df_clustered_profile = df_original_subset.copy()
    df_clustered_profile['Cluster'] = cluster_labels

    cluster_means_numeric = pd.DataFrame()
    cluster_cat_proportions = {}

    # Exclude -1 (noise) from profiling
    clusters_to_profile = [c for c in df_clustered_profile['Cluster'].unique() if c != -1]

    if clusters_to_profile:
        # Numeric Means
        cluster_means_numeric = df_clustered_profile[df_clustered_profile['Cluster'].isin(clusters_to_profile)].groupby('Cluster')[selected_numeric].mean()

        # Categorical Proportions
        for cat_col in selected_categorical:
            if cat_col in df_clustered_profile.columns:
                cat_proportions_df = pd.crosstab(df_clustered_profile['Cluster'], df_clustered_profile[cat_col], normalize='index')
                cluster_cat_proportions[cat_col] = cat_proportions_df[cat_proportions_df.index.isin(clusters_to_profile)] # Filter out noise

    return cluster_means_numeric, cluster_cat_proportions, df_clustered_profile

def get_cluster_description(cluster_id, cluster_mean_values, cat_proportions,
                            overall_numeric_q25, overall_numeric_q75,
                            selected_numeric, selected_categorical):
    """Generates a human-readable description for a single cluster."""
    description_parts = []

    # Numeric feature descriptions
    for col in selected_numeric:
        cluster_val = cluster_mean_values[col]
        q25 = overall_numeric_q25[col]
        q75 = overall_numeric_q75[col]

        val_str = f'{cluster_val:.1f}'
        feature_name = col.lower().replace(' (k$)', '').replace(' (1-100)', '')

        if cluster_val > q75:
            description_parts.append(f'relatively **high {feature_name}** ({val_str})')
        elif cluster_val < q25:
            description_parts.append(f'relatively **low {feature_name}** ({val_str})')
        else:
            description_parts.append(f'average {feature_name} ({val_str})')

    # Categorical feature descriptions
    for cat_col in selected_categorical:
        if cat_col in cat_proportions and cluster_id in cat_proportions[cat_col].index:
            cluster_cat_dist = cat_proportions[cat_col].loc[cluster_id]
            
            # Filter out columns that are not directly relevant to original categories (e.g., if any _NaN columns existed)
            relevant_cat_cols = [c for c in cluster_cat_dist.index if c.startswith(cat_col + '_') or c == cat_col]
            if not relevant_cat_cols: continue

            dominant_category_ohe = cluster_cat_dist[relevant_cat_cols].idxmax()
            dominant_proportion = cluster_cat_dist[dominant_category_ohe] * 100
            original_category_name = dominant_category_ohe.replace(f"{cat_col}_", "").replace("_", " ").lower()

            if dominant_proportion > 75:
                description_parts.append(f'predominantly **{original_category_name}**')
            elif dominant_proportion > 55:
                description_parts.append(f'mostly **{original_category_name}**')
            elif dominant_proportion > 40:
                description_parts.append(f'a significant presence of **{original_category_name}**')

            # Special handling for gender if both categories are substantial
            if cat_col.lower() == 'gender' and len(relevant_cat_cols) > 1:
                other_gender_ohe = [c for c in relevant_cat_cols if c != dominant_category_ohe][0]
                other_proportion = cluster_cat_dist[other_gender_ohe] * 100
                other_original_name = other_gender_ohe.replace(f"{cat_col}_", "").replace("_", " ").lower()
                if other_proportion > 25 and dominant_proportion < 75:
                    description_parts.append(f'with a notable presence of **{other_original_name}**')

    return "This segment is characterized by " + ", ".join(description_parts) + "." if description_parts else "No specific characteristics identified based on available features."


def generate_comprehensive_report(report_settings, df_original_full, df_clustered_output,
                                  pca_plot_bytes, profile_plot_bytes,
                                  cluster_means_numeric, cluster_cat_proportions):
    """
    Generates a comprehensive Word document report.
    """
    document = Document()
    style = document.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    # Title Page
    document.add_heading('Customer Segmentation Report', level=0)
    document.add_paragraph(f'Date: {pd.to_datetime("today").strftime("%Y-%m-%d")}')
    document.add_paragraph('Developed using Unsupervised Machine Learning')
    document.add_paragraph('\n')
    document.add_section(WD_SECTION.NEW_PAGE)

    # 1. Executive Summary
    document.add_heading('1. Executive Summary', level=1)
    document.add_paragraph("This report details the customer segmentation analysis performed on the provided dataset. Using unsupervised machine learning, distinct customer groups have been identified, enabling targeted marketing and business strategies.")
    document.add_paragraph('\n')

    # Automated Executive Summary Snippets
    if not cluster_means_numeric.empty:
        overall_numeric_q25 = df_clustered_output[report_settings['selected_numeric_cols']].quantile(0.25)
        overall_numeric_q75 = df_clustered_output[report_settings['selected_numeric_cols']].quantile(0.75)

        for cluster_id, cluster_mean_values in cluster_means_numeric.iterrows():
            cluster_percentage = (df_clustered_output['Cluster'] == cluster_id).sum() / len(df_clustered_output) * 100
            desc = get_cluster_description(cluster_id, cluster_mean_values, cluster_cat_proportions,
                                           overall_numeric_q25, overall_numeric_q75,
                                           report_settings['selected_numeric_cols'], report_settings['selected_categorical_cols'])
            p = document.add_paragraph()
            p.add_run(f"**Cluster {cluster_id}** ({cluster_percentage:.1f}% of customers): ").bold = True
            # Add the description, handling bolding from markdown-like syntax
            parts = desc.split('**')
            for i, part in enumerate(parts):
                if i % 2 == 1: # If it's an odd index, it's bolded text
                    p.add_run(part).bold = True
                else:
                    p.add_run(part)
            document.add_paragraph('\n')


    # 2. Project Setup & Data Overview
    document.add_heading('2. Project Setup & Data Overview', level=1)
    document.add_heading('2.1 Data Loading & Initial Scan', level=2)
    document.add_paragraph(f'Original dataset loaded with {report_settings["original_shape"][0]} rows and {report_settings["original_shape"][1]} columns.')
    numeric_cols_str = ', '.join([col for col in df_original_full.columns if pd.api.types.is_numeric_dtype(df_original_full[col])])
    object_cols_str = ', '.join([col for col in df_original_full.columns if pd.api.types.is_object_dtype(df_original_full[col])])
    document.add_paragraph(f'Identified Numeric Columns: {numeric_cols_str if numeric_cols_str else "None"}')
    document.add_paragraph(f'Identified Categorical (Object) Columns: {object_cols_str if object_cols_str else "None"}')
    missing_data_info = (df_original_full.isnull().sum() / len(df_original_full) * 100)
    if missing_data_info[missing_data_info > 0].any():
        document.add_paragraph("Missing values were detected in the following columns (percentage):")
        for col, perc in missing_data_info[missing_data_info > 0].items():
            document.add_paragraph(f'- {col}: {perc:.2f}%')
    else:
        document.add_paragraph("No significant missing values were detected in the original dataset.")

    document.add_heading('2.2 Feature Selection & Preprocessing', level=2)
    document.add_paragraph(f'Features selected for clustering: {", ".join(report_settings["selected_features"])}')
    document.add_paragraph(f'Missing data handling strategy: "{report_settings["missing_strategy"].replace("_", " ")}". After handling, the data used for clustering contains {report_settings["rows_after_missing_handling"]} rows.')
    if report_settings['selected_categorical_cols']:
        document.add_paragraph(f'Categorical features ({", ".join(report_settings["selected_categorical_cols"])}) were One-Hot Encoded.')
    else:
        document.add_paragraph("No categorical features were selected for encoding.")
    document.add_paragraph(f'All relevant features were then scaled using StandardScaler.')
    if report_settings['train_ratio'] < 1.0:
        document.add_paragraph(f'Data was split into a training set ({int(report_settings["train_ratio"]*100)}%) and a testing set ({int((1-report_settings["train_ratio"])*100)}%).')
    else:
        document.add_paragraph("No train-test split was applied; all data was used for training.")


    # 3. Clustering Methodology & Evaluation
    document.add_heading('3. Clustering Methodology & Evaluation', level=1)
    document.add_heading('3.1 Algorithm Selection', level=2)
    document.add_paragraph(f'The chosen clustering algorithm for this analysis is: **{report_settings["chosen_algorithm"]}**.')
    if report_settings['n_clusters']:
        document.add_paragraph(f'Number of clusters (k) selected: **{report_settings["n_clusters"]}**.')
    elif report_settings['eps_dbscan'] and report_settings['min_samples_dbscan']:
        document.add_paragraph(f'DBSCAN parameters: eps={report_settings["eps_dbscan"]}, min_samples={report_settings["min_samples_dbscan"]}.')
    document.add_paragraph('The selection was made based on evaluating various algorithms (KMeans, GMM, Agglomerative Clustering) across a range of cluster numbers, using internal validation metrics. Visual plots were used to identify optimal configurations.')

    document.add_heading('3.2 Final Model Performance', level=2)
    document.add_paragraph('The final model\'s performance was assessed using standard internal clustering metrics:')
    document.add_paragraph(f'- **Silhouette Score**: {format_metric(report_settings.get("final_silhouette_score_train"))} (Training Data)')
    document.add_paragraph('  *Measures how similar an object is to its own cluster compared to other clusters. A higher score indicates better-defined clusters, with points closely matched to their own cluster and well-separated from neighboring clusters. Range: -1 to 1.*')
    document.add_paragraph(f'- **Davies-Bouldin Index**: {format_metric(report_settings.get("final_davies_bouldin_score_train"))} (Training Data)')
    document.add_paragraph('  *Indicates the average similarity ratio between each cluster and its most similar cluster. A lower score signifies better clustering, where clusters are compact and well-separated. Range: 0 to infinity.*')
    document.add_paragraph(f'- **Calinski-Harabasz Index**: {format_metric(report_settings.get("final_calinski_harabasz_score_train"))} (Training Data)')
    document.add_paragraph('  *Relates the ratio of between-cluster variance to within-cluster variance. A higher score generally indicates better-defined clusters. Range: 0 to infinity.*')

    if report_settings['train_ratio'] < 1.0 and \
       (report_settings.get('final_silhouette_score_test') is not None or \
        report_settings.get('final_davies_bouldin_score_test') is not None or \
        report_settings.get('final_calinski_harabasz_test') is not None):
        document.add_paragraph(f'\nMetrics on Test Data:')
        document.add_paragraph(f'- **Silhouette Score**: {format_metric(report_settings.get("final_silhouette_score_test"))}')
        document.add_paragraph(f'- **Davies-Bouldin Index**: {format_metric(report_settings.get("final_davies_bouldin_score_test"))}')
        document.add_paragraph(f'- **Calinski-Harabasz Index**: {format_metric(report_settings.get("final_calinski_harabasz_test"))}')
    elif report_settings['train_ratio'] < 1.0:
         document.add_paragraph(f'\nNote: Test set evaluation metrics were not applicable or calculated for the chosen algorithm.')

    if report_settings['chosen_algorithm'].lower() == 'dbscan' and 'n_noise_points_train' in report_settings:
        document.add_paragraph(f'\nDBSCAN identified {report_settings["n_noise_points_train"]} data points as noise (unassigned to any cluster) in the training data.')


    # 4. Customer Segment Profiles
    document.add_heading('4. Customer Segment Profiles', level=1)
    document.add_heading('4.1 Cluster Distribution', level=2)
    
    # Calculate cluster counts and percentages for the report
    valid_labels = df_clustered_output['Cluster'].values
    if -1 in np.unique(valid_labels) and report_settings['chosen_algorithm'].lower() == 'dbscan':
        cluster_counts = pd.Series(valid_labels).value_counts().drop(labels=[-1], errors='ignore').sort_index()
        total_non_noise = cluster_counts.sum()
        cluster_percentages = (cluster_counts / total_non_noise * 100).round(2) if total_non_noise > 0 else pd.Series(dtype=float)
        noise_count = list(valid_labels).count(-1)
        document.add_paragraph(f"A total of {len(cluster_counts)} distinct customer clusters were identified.")
        document.add_paragraph(f"Note: {noise_count} data points were identified as noise and are not assigned to any cluster.")
    else:
        cluster_counts = pd.Series(valid_labels).value_counts().sort_index()
        total_customers = cluster_counts.sum()
        cluster_percentages = (cluster_counts / total_customers * 100).round(2) if total_customers > 0 else pd.Series(dtype=float)
        document.add_paragraph(f"A total of {len(cluster_counts)} distinct customer clusters were identified.")

    # Add distribution table
    table = document.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Cluster ID'
    hdr_cells[1].text = 'Number of Customers'
    hdr_cells[2].text = 'Percentage (%)'
    for cluster_id in cluster_counts.index:
        row_cells = table.add_row().cells
        row_cells[0].text = str(cluster_id)
        row_cells[1].text = str(cluster_counts[cluster_id])
        row_cells[2].text = str(cluster_percentages[cluster_id])
    document.add_paragraph('\n')

    document.add_heading('4.2 Cluster Characterization', level=2)
    document.add_paragraph('Below are the characteristics of each identified customer segment based on their average feature values.')

    # Get overall statistics for comparison (from the original df_clustered_output, which has original features)
    overall_numeric_q25 = df_clustered_output[report_settings['selected_numeric_cols']].quantile(0.25)
    overall_numeric_q75 = df_clustered_output[report_settings['selected_numeric_cols']].quantile(0.75)

    if not cluster_means_numeric.empty:
        for cluster_id, cluster_mean_values in cluster_means_numeric.iterrows():
            if cluster_id == -1: continue # Skip noise cluster

            cluster_percentage = (df_clustered_output['Cluster'] == cluster_id).sum() / len(df_clustered_output) * 100
            desc = get_cluster_description(cluster_id, cluster_mean_values, cluster_cat_proportions,
                                           overall_numeric_q25, overall_numeric_q75,
                                           report_settings['selected_numeric_cols'], report_settings['selected_categorical_cols'])
            p = document.add_paragraph()
            p.add_run(f"**Cluster {cluster_id}** ({cluster_percentage:.1f}% of customers): ").bold = True
            # Add the description, handling bolding from markdown-like syntax
            parts = desc.split('**')
            for i, part in enumerate(parts):
                if i % 2 == 1: # If it's an odd index, it's bolded text
                    p.add_run(part).bold = True
                else:
                    p.add_run(part)
            document.add_paragraph('\n')
    else:
        document.add_paragraph("No cluster profiles generated for characterization.")


    document.add_heading('4.3 Detailed Numeric Feature Profiles', level=2)
    document.add_paragraph('The table below presents the average values for each numeric feature within each cluster, providing a quantitative basis for the segment descriptions.')
    if not cluster_means_numeric.empty:
        num_clusters = len(cluster_means_numeric)
        num_features = len(cluster_means_numeric.columns)
        table = document.add_table(rows=num_clusters + 1, cols=num_features + 1)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Cluster ID'
        for i, col in enumerate(cluster_means_numeric.columns):
            hdr_cells[i+1].text = col
        for r, (cluster_id, row_data) in enumerate(cluster_means_numeric.iterrows()):
            row_cells = table.rows[r+1].cells
            row_cells[0].text = str(cluster_id)
            for c, col in enumerate(cluster_means_numeric.columns):
                row_cells[c+1].text = f'{row_data[col]:.2f}'
    else:
        document.add_paragraph("No numeric feature profile data available to display.")
    document.add_paragraph('\n')

    document.add_heading('4.4 Detailed Categorical Feature Proportions', level=2)
    document.add_paragraph('This section details the distribution of categorical features within each cluster, showing the percentage of customers belonging to each category.')
    if report_settings['selected_categorical_cols']:
        for cat_col in report_settings['selected_categorical_cols']:
            if cat_col in cluster_cat_proportions:
                cat_proportions_df = cluster_cat_proportions[cat_col]
                document.add_heading(f'Proportions for: {cat_col}', level=3)
                num_clusters = len(cat_proportions_df)
                num_categories = len(cat_proportions_df.columns)
                table = document.add_table(rows=num_clusters + 1, cols=num_categories + 1)
                table.style = 'Table Grid'
                hdr_cells = table.rows[0].cells
                hdr_cells[0].text = 'Cluster ID'
                for i, col_name in enumerate(cat_proportions_df.columns):
                    clean_col_name = col_name.replace(f"{cat_col}_", "").replace("_", " ").title()
                    hdr_cells[i+1].text = clean_col_name
                for r, (cluster_id, row_data) in enumerate(cat_proportions_df.iterrows()):
                    row_cells = table.rows[r+1].cells
                    row_cells[0].text = str(cluster_id)
                    for c, col_name in enumerate(cat_proportions_df.columns):
                        row_cells[c+1].text = f'{row_data[col_name]*100:.1f}%'
                document.add_paragraph('\n')
            else:
                document.add_paragraph(f"No proportion data available for categorical feature: {cat_col}.")
    else:
        document.add_paragraph("No categorical features were selected for profiling.")
    document.add_paragraph('\n')

    # 5. Visualizations
    document.add_heading('5. Visualizations', level=1)
    document.add_paragraph('This section presents key visualizations that illustrate the identified customer segments.')

    document.add_heading('5.1 Customer Clusters (PCA-Reduced)', level=2)
    if pca_plot_bytes:
        document.add_picture(io.BytesIO(pca_plot_bytes), width=Inches(6))
        document.add_paragraph('\n')
        document.add_paragraph(
            f'**What it shows:** This scatter plot visualizes the customer data reduced to two principal components (PC1 and PC2). '
            f'These components capture the most variance in your original features, allowing us to see how the clusters '
            f'are separated in a 2D space. Each point represents a customer, colored according to their assigned cluster.'
        )
        document.add_paragraph(
            f'**Key Findings:** In this plot, we observe that the **{report_settings["n_clusters"]} clusters** identified by '
            f'{report_settings["chosen_algorithm"]} appear generally distinct, with some areas of overlap. '
            f'This visual separation supports the model\'s ability to group similar customers together.'
        )
    else:
        document.add_paragraph("PCA cluster plot not generated or available.")
    document.add_paragraph('\n')

    document.add_heading('5.2 Average Feature Values per Cluster', level=2)
    if profile_plot_bytes:
        document.add_picture(io.BytesIO(profile_plot_bytes), width=Inches(6))
        document.add_paragraph('\n')
        document.add_paragraph(
            f'**What it shows:** These bar charts display the average value for each selected numeric feature '
            f'across all identified customer clusters. '
            f'This allows for a direct comparison of how each feature varies from one segment to another, '
            f'providing concrete insights into what defines each group.'
        )
        document.add_paragraph(
            f'**Key Findings:** These plots clearly illustrate the distinct characteristics of each cluster. '
            f'Refer to the "Cluster Characterization" section above for detailed descriptions of each segment\'s defining features.'
        )
    else:
        document.add_paragraph("Cluster profile plot not generated or available.")
    document.add_paragraph('\n')

    # 6. Limitations and Future Work
    document.add_heading('6. Limitations and Future Work', level=1)
    document.add_paragraph(
        '**Limitations:** This customer segmentation analysis, while insightful, has certain limitations. '
        'The choice of the number of clusters (k) or DBSCAN parameters (eps, min_samples) is often subjective and '
        'relies on internal validation metrics which do not directly measure business impact. '
        'The quality of segments is dependent on the input features; adding more relevant data (e.g., online behavior, purchase history specifics) '
        'could lead to more granular and actionable segments. Furthermore, unsupervised learning does not guarantee '
        'that the clusters are inherently "correct" in a business sense; their utility must be validated through practical application.'
    )
    document.add_paragraph('\n')
    document.add_paragraph(
        '**Recommendations & Future Work:** Based on these customer segments, the following actions are recommended:'
    )
    document.add_paragraph(
        '•   **Targeted Marketing Campaigns:** Design specific marketing messages, promotions, and product recommendations '
        'tailored to the unique characteristics of each cluster.'
    )
    document.add_paragraph(
        '•   **Personalized Customer Experience:** Adapt customer service approaches and in-store/online experiences to better '
        'serve the preferences of each segment. '
    )
    document.add_paragraph(
        '•   **Product Development:** Identify unmet needs or opportunities for new product development by analyzing the '
        'characteristics of underserved or highly distinct segments.'
    )
    document.add_paragraph(
        '•   **Monitor Segment Evolution:** Regularly re-run the clustering analysis to monitor if customer behaviors or '
        'segment compositions change over time, ensuring strategies remain relevant.'
    )
    document.add_paragraph(
        '•   **A/B Testing:** Implement A/B tests for different strategies on different segments to empirically measure '
        'the effectiveness of the segmentation.'
    )
    document.add_paragraph('\n')

    # Save to BytesIO for download
    bio = io.BytesIO()
    document.save(bio)
    bio.seek(0)
    return bio.getvalue()


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Customer Segmentation Dashboard")

st.title("📊 Interactive Customer Segmentation Dashboard")
st.markdown("""
    Upload your customer data (CSV/Excel), select relevant features, choose an unsupervised clustering algorithm,
    and discover meaningful customer segments. Get insights through interactive visualizations and a comprehensive downloadable report.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Upload Data & Configure")

uploaded_file = st.sidebar.file_uploader("Upload your data file (.csv or .xlsx)", type=["csv", "xlsx"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}. Please ensure it's a valid CSV or XLSX.")

if df is not None:
    st.header("1. Data Overview")
    st.write("First 5 rows of your uploaded data:")
    st.dataframe(df.head())
    st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    st.header("2. Feature Selection & Preprocessing")
    all_columns = df.columns.tolist()

    # Suggest default features based on common patterns (can be improved with LLM in future)
    numeric_cols_detected = [col for col in all_columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols_detected = [col for col in all_columns if pd.api.types.is_object_dtype(df[col])]

    # Exclude common IDs and target variables by default
    suggested_numeric_default = [col for col in numeric_cols_detected if 'id' not in col.lower() and 'num' not in col.lower() and 'flag' not in col.lower() and 'bayes' not in col.lower()]
    suggested_categorical_default = [col for col in categorical_cols_detected if 'flag' not in col.lower() and 'id' not in col.lower()]


    selected_numeric_cols = st.multiselect(
        "Select Numeric Features for Clustering:",
        options=numeric_cols_detected,
        default=suggested_numeric_default
    )
    selected_categorical_cols = st.multiselect(
        "Select Categorical Features for Clustering (will be One-Hot Encoded):",
        options=categorical_cols_detected,
        default=suggested_categorical_default
    )

    missing_strategy = st.selectbox(
        "How to handle missing values in selected features?",
        ("drop_rows", "impute"),
        format_func=lambda x: x.replace("_", " ").title()
    )

    # Train-test split is less common for pure unsupervised, but keep the option
    train_ratio = st.slider("Train-Test Split Ratio (0 for no split):", min_value=0.0, max_value=0.9, value=0.0, step=0.1)

    if not selected_numeric_cols and not selected_categorical_cols:
        st.warning("Please select at least one feature (numeric or categorical) for clustering.")
    else:
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            try:
                scaled_df, df_for_profiling, rows_dropped_count, features_after_encoding = \
                    preprocess_data(df, selected_numeric_cols, selected_categorical_cols, missing_strategy)

                st.success("Data Preprocessing Complete!")
                st.write(f"Rows dropped due to missing values: {rows_dropped_count}")
                st.write("First 5 rows of processed (scaled and encoded) data:")
                st.dataframe(scaled_df.head())

            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
                st.stop() # Stop execution if preprocessing fails

        st.header("3. Optimal K & Algorithm Selection")
        k_range_eval = range(2, min(11, len(scaled_df))) # Max k is min(10, num_samples-1)

        if len(scaled_df) < 2:
            st.warning("Not enough data points to perform clustering evaluation (need at least 2).")
        else:
            with st.spinner(f"Evaluating algorithms for k from {min(k_range_eval)} to {max(k_range_eval)}..."):
                kmeans_scores, gmm_scores, agglo_scores = evaluate_algorithms(scaled_df, k_range_eval)

            # Plotting the evaluation metrics
            fig_eval, axes_eval = plt.subplots(3, 1, figsize=(10, 15))
            fig_eval.suptitle('Clustering Evaluation Metrics for Different K Values', fontsize=16)

            # Silhouette Score (Higher is better)
            axes_eval[0].plot(k_range_eval[:len(kmeans_scores['Silhouette'])], kmeans_scores['Silhouette'], marker='o', label='KMeans')
            axes_eval[0].plot(k_range_eval[:len(gmm_scores['Silhouette'])], gmm_scores['Silhouette'], marker='x', label='GMM')
            axes_eval[0].plot(k_range_eval[:len(agglo_scores['Silhouette'])], agglo_scores['Silhouette'], marker='s', label='Agglomerative')
            axes_eval[0].set_title('Silhouette Score (Higher is Better)')
            axes_eval[0].set_xlabel('Number of Clusters (k)')
            axes_eval[0].set_ylabel('Score')
            axes_eval[0].legend()
            axes_eval[0].grid(True)

            # Davies-Bouldin Index (Lower is better)
            axes_eval[1].plot(k_range_eval[:len(kmeans_scores['Davies-Bouldin'])], kmeans_scores['Davies-Bouldin'], marker='o', label='KMeans')
            axes_eval[1].plot(k_range_eval[:len(gmm_scores['Davies-Bouldin'])], gmm_scores['Davies-Bouldin'], marker='x', label='GMM')
            axes_eval[1].plot(k_range_eval[:len(agglo_scores['Davies-Bouldin'])], agglo_scores['Davies-Bouldin'], marker='s', label='Agglomerative')
            axes_eval[1].set_title('Davies-Bouldin Index (Lower is Better)')
            axes_eval[1].set_xlabel('Number of Clusters (k)')
            axes_eval[1].set_ylabel('Index')
            axes_eval[1].legend()
            axes_eval[1].grid(True)

            # Calinski-Harabasz Index (Higher is better)
            axes_eval[2].plot(k_range_eval[:len(kmeans_scores['Calinski-Harabasz'])], kmeans_scores['Calinski-Harabasz'], marker='o', label='KMeans')
            axes_eval[2].plot(k_range_eval[:len(gmm_scores['Calinski-Harabasz'])], gmm_scores['Calinski-Harabasz'], marker='x', label='GMM')
            axes_eval[2].plot(k_range_eval[:len(agglo_scores['Calinski-Harabasz'])], agglo_scores['Calinski-Harabasz'], marker='s', label='Agglomerative')
            axes_eval[2].set_title('Calinski-Harabasz Index (Higher is Better)')
            axes_eval[2].set_xlabel('Number of Clusters (k)')
            axes_eval[2].set_ylabel('Index')
            axes_eval[2].legend()
            axes_eval[2].grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            st.pyplot(fig_eval)

        st.sidebar.subheader("Final Algorithm & Parameters")
        chosen_algorithm = st.sidebar.selectbox(
            "Choose Clustering Algorithm:",
            ("KMeans", "Gaussian Mixture Model", "Agglomerative Clustering", "DBSCAN")
        )

        n_clusters = None
        eps_dbscan = None
        min_samples_dbscan = None

        if chosen_algorithm in ["KMeans", "Gaussian Mixture Model", "Agglomerative Clustering"]:
            n_clusters = st.sidebar.slider("Number of Clusters (k):", min_value=2, max_value=min(10, len(scaled_df)-1), value=min(5, len(scaled_df)-1))
            if n_clusters < 2 and len(scaled_df) >= 2:
                st.sidebar.warning("Number of clusters must be at least 2.")
                st.stop()
            elif len(scaled_df) < 2:
                st.sidebar.warning("Not enough data points for clustering.")
                st.stop()

        elif chosen_algorithm == "DBSCAN":
            eps_dbscan = st.sidebar.slider("DBSCAN epsilon (eps):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            min_samples_dbscan = st.sidebar.slider("DBSCAN min_samples:", min_value=2, max_value=20, value=max(5, 2 * len(features_after_encoding)))

        # --- Run Clustering Button ---
        if st.button("Run Clustering & Generate Report"):
            if n_clusters is None and chosen_algorithm != "DBSCAN":
                st.error("Please select a valid number of clusters.")
            elif chosen_algorithm == "DBSCAN" and (eps_dbscan is None or min_samples_dbscan is None):
                st.error("Please set DBSCAN parameters.")
            else:
                st.header("4. Clustering Results")
                with st.spinner(f"Running {chosen_algorithm} clustering..."):
                    model, cluster_labels, n_noise_points = train_clustering_model(
                        scaled_df, chosen_algorithm, n_clusters, eps_dbscan, min_samples_dbscan
                    )

                if cluster_labels is not None:
                    st.success(f"Clustering complete! Identified {len(np.unique(cluster_labels)) if -1 not in np.unique(cluster_labels) else len(np.unique(cluster_labels)) -1} clusters.")
                    if chosen_algorithm == "DBSCAN" and n_noise_points > 0:
                        st.write(f"DBSCAN identified {n_noise_points} noise points (labeled -1).")

                    # Generate profiles
                    cluster_means_numeric, cluster_cat_proportions, df_clustered_output = \
                        generate_cluster_profiles(df_for_profiling, cluster_labels, selected_numeric_cols, selected_categorical_cols)

                    # --- Evaluation Metrics ---
                    st.subheader("4.1 Evaluation Metrics")
                    # Calculate metrics on the full clustered data (scaled_df)
                    if len(np.unique(cluster_labels)) > 1: # Need at least 2 clusters for metrics
                        final_silhouette = silhouette_score(scaled_df, cluster_labels)
                        final_davies_bouldin = davies_bouldin_score(scaled_df, cluster_labels)
                        final_calinski_harabasz = calinski_harabasz_score(scaled_df, cluster_labels)
                        st.write(f"- **Silhouette Score**: {format_metric(final_silhouette)}")
                        st.write(f"- **Davies-Bouldin Index**: {format_metric(final_davies_bouldin)}")
                        st.write(f"- **Calinski-Harabasz Index**: {format_metric(final_calinski_harabasz)}")
                    else:
                        st.warning("Cannot compute standard metrics (less than 2 clusters or only noise found).")


                    # --- Cluster Profiles ---
                    st.subheader("4.2 Cluster Profiles")
                    if not cluster_means_numeric.empty:
                        st.write("#### Average Numeric Features per Cluster:")
                        st.dataframe(cluster_means_numeric.round(2))

                    if selected_categorical_cols:
                        for cat_col in selected_categorical_cols:
                            if cat_col in cluster_cat_proportions:
                                st.write(f"#### Proportions for {cat_col} per Cluster:")
                                st.dataframe((cluster_cat_proportions[cat_col]*100).round(1))

                    # --- Visualizations ---
                    st.subheader("4.3 Visualizations")

                    # PCA Plot
                    if scaled_df.shape[1] > 1 and len(np.unique(cluster_labels)) > 1: # PCA needs at least 2 features and 2 clusters
                        pca = PCA(n_components=2, random_state=42)
                        principal_components = pca.fit_transform(scaled_df)
                        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=scaled_df.index)
                        pca_df['Cluster'] = cluster_labels

                        fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
                        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', ax=ax_pca, legend='full', s=100, alpha=0.8)
                        ax_pca.set_title(f'Customer Clusters (PCA-Reduced) - {chosen_algorithm} (k={n_clusters if n_clusters else "N/A"})')
                        ax_pca.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
                        ax_pca.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
                        ax_pca.grid(True)
                        st.pyplot(fig_pca)

                        # Save PCA plot to bytes for report
                        pca_plot_buffer = io.BytesIO()
                        fig_pca.savefig(pca_plot_buffer, format='png', bbox_inches='tight')
                        pca_plot_bytes = pca_plot_buffer.getvalue()
                        plt.close(fig_pca) # Close figure to free memory
                    else:
                        st.info("PCA plot requires at least 2 features and more than 1 cluster.")
                        pca_plot_bytes = None

                    # Cluster Profile Bar Plots
                    if not cluster_means_numeric.empty:
                        num_numeric_features = len(selected_numeric_cols)
                        if num_numeric_features > 0:
                            fig_profile, axes_profile = plt.subplots(num_numeric_features, 1, figsize=(10, 4 * num_numeric_features))
                            fig_profile.suptitle('Average Feature Values per Cluster (Original Scale)', y=1.02, fontsize=16)

                            if num_numeric_features == 1:
                                axes_profile = [axes_profile]

                            for i, col in enumerate(selected_numeric_cols):
                                sns.barplot(x=cluster_means_numeric.index, y=cluster_means_numeric[col], ax=axes_profile[i], palette='viridis')
                                axes_profile[i].set_title(f'Average {col}')
                                axes_profile[i].set_xlabel('Cluster')
                                axes_profile[i].set_ylabel(col)
                                axes_profile[i].grid(axis='y')

                            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                            st.pyplot(fig_profile)

                            # Save profile plot to bytes for report
                            profile_plot_buffer = io.BytesIO()
                            fig_profile.savefig(profile_plot_buffer, format='png', bbox_inches='tight')
                            profile_plot_bytes = profile_plot_buffer.getvalue()
                            plt.close(fig_profile) # Close figure to free memory
                        else:
                            st.info("No numeric features selected to generate cluster profile plots.")
                            profile_plot_bytes = None
                    else:
                        profile_plot_bytes = None


                    # --- Download Buttons ---
                    st.subheader("5. Downloads")

                    # Prepare report settings for the report function
                    report_settings = {
                        'original_shape': df.shape,
                        'selected_features': selected_numeric_cols + selected_categorical_cols,
                        'selected_numeric_cols': selected_numeric_cols,
                        'selected_categorical_cols': selected_categorical_cols,
                        'missing_strategy': missing_strategy,
                        'rows_after_missing_handling': scaled_df.shape[0],
                        'train_ratio': train_ratio, # This is not used in the report, but kept for consistency
                        'chosen_algorithm': chosen_algorithm,
                        'n_clusters': n_clusters,
                        'eps_dbscan': eps_dbscan,
                        'min_samples_dbscan': min_samples_dbscan,
                        'final_silhouette_score_train': final_silhouette if 'final_silhouette' in locals() else None,
                        'final_davies_bouldin_score_train': final_davies_bouldin if 'final_davies_bouldin' in locals() else None,
                        'final_calinski_harabasz_score_train': final_calinski_harabasz if 'final_calinski_harabasz' in locals() else None,
                        # Test scores are not calculated in this Streamlit app for simplicity, as we use full data for training
                        'n_noise_points_train': n_noise_points,
                    }

                    # Generate and Download Report
                    with st.spinner("Generating comprehensive report..."):
                        report_bytes = generate_comprehensive_report(
                            report_settings, df, df_clustered_output,
                            pca_plot_bytes, profile_plot_bytes,
                            cluster_means_numeric, cluster_cat_proportions
                        )
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
                    st.error("Clustering failed. Please check your data and selected parameters.")

else:
    st.info("Please upload a data file (.csv or .xlsx) in the sidebar to begin the analysis.")
