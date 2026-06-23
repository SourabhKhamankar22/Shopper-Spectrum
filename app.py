import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Load Machine Learning Assets (Cached for speed)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        kmeans_model = joblib.load('kmeans_model.joblib')
        scaler = joblib.load('rfm_scaler.joblib')
        knn_model = joblib.load('knn_recommender.joblib')
        pivot_table = pd.read_pickle('pivot_table.joblib')
        return kmeans_model, scaler, knn_model, pivot_table
    except Exception as e:
        st.error(f"⚠️ Error loading model files. Please ensure 'kmeans_model.joblib', 'rfm_scaler.joblib', 'knn_recommender.joblib', and 'pivot_table.pkl' are in the same folder as this script.\n\nDetails: {e}")
        return None, None, None, None

kmeans_model, scaler, knn_model, pivot_table = load_models()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3144/3144456.png", width=100)
st.sidebar.title("Shopper Spectrum")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Module:", ["🛒 Product Recommendation", "👥 Customer Segmentation"])
st.sidebar.markdown("---")
st.sidebar.info("Developed for E-Commerce Retail Analytics.")

# ---------------------------------------------------------
# MODULE 1: Product Recommendation
# ---------------------------------------------------------
if app_mode == "🛒 Product Recommendation":
    st.title("🛒 Product Recommendation Engine")
    st.markdown("Enter a product name to discover exactly what your customers frequently buy alongside it.")
    
    st.markdown("### Search Product")
    
    # Text input box for Product Name
    product_input = st.text_input("Enter Product Name (e.g., 'white hanging heart t-light holder'):", "")
    
    if st.button("Get Recommendations"):
        if not product_input:
            st.warning("Please enter a product name.")
        elif pivot_table is not None:
            # Standardize input for searching
            search_term = product_input.strip().lower()
            
            # Check for exact match
            exact_match = None
            for prod in pivot_table.index:
                if str(prod).strip().lower() == search_term:
                    exact_match = prod
                    break
            
            if exact_match:
                # 1. Get the vector for the product
                product_idx = pivot_table.index.get_loc(exact_match)
                product_vector = pivot_table.iloc[product_idx, :].values.reshape(1, -1)
                
                # 2. Predict Nearest Neighbors
                distances, indices = knn_model.kneighbors(product_vector)
                
                # 3. Display as styled cards
                st.markdown(f"### Top 5 Recommendations for: **{exact_match.title()}**")
                
                # Create 5 columns for a card-like view
                cols = st.columns(5)
                
                # We start at index 1 to skip the searched product itself (index 0)
                for i in range(1, 6):
                    if i < len(indices.flatten()):
                        rec_item = pivot_table.index[indices.flatten()[i]]
                        rec_dist = distances.flatten()[i]
                        
                        # Displaying in columns
                        with cols[i-1]:
                            st.info(f"**Recommendation {i}**\n\n🛍️ {rec_item.title()}\n\n*(Similarity Distance: {rec_dist:.2f})*")
                            
            else:
                # If no exact match, suggest partial matches
                st.error(f"❌ Product '{product_input}' not found in the database.")
                partial_matches = [str(prod) for prod in pivot_table.index if search_term in str(prod).lower()]
                
                if partial_matches:
                    st.warning("Did you mean one of these?")
                    for match in partial_matches[:5]: # Show up to 5 suggestions
                        st.write(f"- {match}")

# ---------------------------------------------------------
# MODULE 2: Customer Segmentation
# ---------------------------------------------------------
elif app_mode == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation Predictor")
    st.markdown("Input a customer's RFM metrics below to instantly predict their behavioral segment and risk profile.")
    
    st.markdown("### Input Customer Behavior")
    
    # 3 Number inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (in days)", min_value=0, value=15, help="Days since the customer's last purchase.")
    with col2:
        frequency = st.number_input("Frequency (purchases)", min_value=1, value=5, help="Total number of separate transactions.")
    with col3:
        monetary = st.number_input("Monetary (total spend $)", min_value=0.0, value=250.50, help="Total amount of money spent by the customer.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Cluster", type="primary"):
        if kmeans_model is not None and scaler is not None:
            
            # 1. Prepare data into a DataFrame
            unseen_data = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})
            
            # 2. Apply the exact same transformations as training (Log1p -> Scale)
            unseen_log = np.log1p(unseen_data)
            unseen_scaled = scaler.transform(unseen_log)
            
            # 3. Predict Cluster
            pred_cluster = kmeans_model.predict(unseen_scaled)[0]
            
            # 4. Map cluster number to Business Label
            # Standard map based on 4-cluster business requirements
            segment_map_4 = {0: 'At-Risk 📉', 1: 'Regular 🛒', 2: 'High-Value 💎', 3: 'Occasional 🤔'}
            
            # If the optimized model fell back to 2 clusters
            segment_map_2 = {0: 'At-Risk / Occasional 📉', 1: 'High-Value / Regular 💎'}
            
            if hasattr(kmeans_model, 'n_clusters') and kmeans_model.n_clusters == 2:
                segment_name = segment_map_2.get(pred_cluster, f"Cluster {pred_cluster}")
            else:
                segment_name = segment_map_4.get(pred_cluster, f"Cluster {pred_cluster}")
                
            # 5. Display Result
            st.markdown("---")
            st.markdown("### Prediction Result")
            st.success(f"This customer belongs to the segment: **{segment_name}**")
            
            # Actionable advice based on the segment (Textual inference)
            if "High-Value" in segment_name:
                st.write("**Action Plan:** Assign a VIP account manager and offer early access to new product lines to ensure loyalty.")
            elif "At-Risk" in segment_name:
                st.write("**Action Plan:** Trigger an automated win-back email campaign with a heavy discount code (e.g., 20% off) to prevent churn.")
            elif "Regular" in segment_name:
                st.write("**Action Plan:** Utilize the Recommendation Engine to cross-sell products and increase their Average Order Value.")
            else:
                st.write("**Action Plan:** Send nurturing campaigns and standard promotional newsletters to increase their frequency of visits.")