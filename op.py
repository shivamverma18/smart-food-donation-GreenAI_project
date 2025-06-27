# Smart Food Donation System using KMeans Clustering
# --------------------------------------------------
# This Python script is designed to run with Streamlit.
# It clusters incoming food donation offers to the most suitable centers based on current needs.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ----------------------------
# Initial Center Setup
# ----------------------------
# Each center has initial capacity for different food items.
centers = {
    'Center_1': {'roti': 40, 'rice': 30, 'daal': 20, 'others': 10},
    'Center_2': {'roti': 25, 'rice': 40, 'daal': 15, 'others': 5},
    'Center_3': {'roti': 50, 'rice': 20, 'daal': 30, 'others': 10},
    'Center_4': {'roti': 35, 'rice': 25, 'daal': 25, 'others': 5},
    'Center_5': {'roti': 30, 'rice': 35, 'daal': 10, 'others': 15},
    'Center_6': {'roti': 20, 'rice': 20, 'daal': 20, 'others': 20}
}

# Convert dictionary to DataFrame for ML operations
center_df = pd.DataFrame.from_dict(centers, orient='index')
center_df.reset_index(inplace=True)
center_df.rename(columns={'index': 'center'}, inplace=True)

# ----------------------------
# Streamlit Web Interface
# ----------------------------
# User-facing form to accept donation input
st.set_page_config(page_title="Food Donation Clustering", layout="centered")
st.title("🍱 Smart Food Donation - Nagpur")
st.markdown("Help reduce food waste and fight hunger by donating your excess food. We'll route it to the most suitable center.")

# Input form
st.subheader("Enter Your Details")
address = st.text_input("Your Address")
roti = st.number_input("Roti (qty)", min_value=0, value=0)
rice = st.number_input("Rice (qty)", min_value=0, value=0)
daal = st.number_input("Daal/Curry (qty)", min_value=0, value=0)
others = st.number_input("Others (qty)", min_value=0, value=0)

# ----------------------------
# On Submission: Clustering & Assignment
# ----------------------------
if st.button("Submit Donation"):
    # Prepare the input vector from user
    user_input = {'roti': roti, 'rice': rice, 'daal': daal, 'others': others}
    user_vector = np.array([[roti, rice, daal, others]])

    # Perform KMeans clustering on center needs (not location)
    X = center_df[['roti', 'rice', 'daal', 'others']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    # Identify cluster for the current donation input
    user_cluster = kmeans.predict(user_vector)[0]
    cluster_centers = center_df[kmeans.labels_ == user_cluster].copy()

    # ----------------------------
    # Match user to a valid center within the same cluster
    # ----------------------------
    assigned_center = None
    for _, row in cluster_centers.iterrows():
        cname = row['center']
        # Check if all food types can be accepted by the center
        has_capacity = all(user_input[item] <= centers[cname][item] for item in user_input)
        if has_capacity:
            assigned_center = cname
            break

    # ----------------------------
    # Update Center and Show Result
    # ----------------------------
    if assigned_center:
        # Update the center's capacity
        for item in user_input:
            centers[assigned_center][item] -= user_input[item]

        st.success(f"✅ Your donation has been accepted!\n\n**Address:** {address}\n**Assigned Center:** {assigned_center}")
        st.write("### Updated Center Capacity:")
        st.json(centers[assigned_center])
    else:
        st.error("❌ Sorry, no center in the matching cluster can accept your food right now.")