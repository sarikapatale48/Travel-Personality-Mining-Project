import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.decomposition import PCA


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Travel Personality Mining",
    page_icon="✈️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-top: 20px;
}

.profile-title {
    font-size: 30px;
    font-weight: 700;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA AND MODELS
# =========================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(
        "dataset/travel_review.csv"
    )

    # Remove first column (User / ID column)
    df = df.drop(columns=[df.columns[0]])

    # Convert values to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove missing rows
    df = df.dropna()

    return df


@st.cache_resource
def load_models():

    model = joblib.load(
        "models/kmeans_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    return model, scaler


df = load_dataset()
model, scaler = load_models()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("✈️ Travel Personality")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🧭 Find My Travel Profile",
        "📊 Cluster Analysis",
        "📈 Dataset Overview"
    ]
)


# =========================================================
# PERSONALITY INFORMATION
# =========================================================

personality_info = {

    0: {
        "name": "🏛️ Heritage & Sightseeing Traveller",
        "description":
            "This profile shows stronger interest in monuments, "
            "view points, museums and sightseeing locations."
    },

    1: {
        "name": "🌳 Recreation & Culture Traveller",
        "description":
            "This profile shows stronger interest in parks, "
            "theatres, museums and recreational destinations."
    },

    2: {
        "name": "🍔 Food & Stay Explorer",
        "description":
            "This profile shows stronger interest in hotels, "
            "food outlets, juice bars and related destinations."
    },

    3: {
        "name": "🍽️ Entertainment & Social Traveller",
        "description":
            "This profile shows stronger interest in restaurants, "
            "malls, pubs/bars and entertainment destinations."
    },

    4: {
        "name": "🏊 Active & Entertainment Traveller",
        "description":
            "This profile shows stronger interest in swimming pools, "
            "gyms, malls and dance clubs."
    },

    5: {
        "name": "🌿 Nature & Relaxation Traveller",
        "description":
            "This profile shows stronger interest in view points, "
            "gardens, resorts, monuments and beaches."
    }
}


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">✈️ Travel Personality Mining</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Discover traveller profiles from travel rating patterns using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Tourist Records",
            len(df)
        )

    with col2:
        st.metric(
            "Travel Features",
            len(df.columns)
        )

    with col3:
        st.metric(
            "Traveller Clusters",
            model.n_clusters
        )

    st.divider()

    st.header("🎯 Project Objective")

    st.write(
        """
        The objective of this project is to identify different types of
        travellers by analysing their travel-related ratings.

        Instead of manually assigning categories, Machine Learning is used
        to discover groups of tourists having similar rating patterns.
        """
    )

    st.header("🤖 Machine Learning Approach")

    st.write(
        """
        The project uses **K-Means Clustering**, an unsupervised machine
        learning algorithm.

        The main process is:

        Dataset → Data Cleaning → Standardization →
        K-Means Clustering → Traveller Groups →
        Profile Interpretation
        """
    )

    st.info(
        "The profiles shown by this application represent travel preference "
        "patterns. They should not be interpreted as psychological personality types."
    )


# =========================================================
# FIND MY TRAVEL PROFILE
# =========================================================

elif page == "🧭 Find My Travel Profile":

    st.markdown(
        '<div class="main-title">🧭 Find My Travel Profile</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Rate your interest in different travel destinations from 1 to 5."
    )

    st.info(
        "Tip: To see a stronger difference between profiles, change several "
        "related ratings together instead of changing only one slider."
    )

    # -----------------------------------------------------
    # Create input values
    # -----------------------------------------------------

    input_values = {}

    # Two columns for sliders
    col1, col2 = st.columns(2)

    for index, feature in enumerate(df.columns):

        if index % 2 == 0:
            with col1:

                input_values[feature] = st.slider(
                    feature,
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1
                )

        else:
            with col2:

                input_values[feature] = st.slider(
                    feature,
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1
                )

    st.divider()

    # -----------------------------------------------------
    # Prediction button
    # -----------------------------------------------------

    if st.button(
        "🔍 Find My Travel Profile",
        use_container_width=True
    ):

        # Create dataframe in EXACT same column order
        user_df = pd.DataFrame(
            [[input_values[col] for col in df.columns]],
            columns=df.columns
        )

        # Scale input
        user_scaled = scaler.transform(user_df)

        # Predict cluster
        predicted_cluster = int(
            model.predict(user_scaled)[0]
        )

        # Calculate distances to all clusters
        distances = model.transform(user_scaled)[0]

        # Get profile information
        profile = personality_info.get(
            predicted_cluster,
            {
                "name": f"Traveller Profile {predicted_cluster}",
                "description":
                    "A traveller profile discovered from the rating patterns."
            }
        )

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="profile-title">'
            + profile["name"]
            + '</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Assigned Cluster:** {predicted_cluster}"
        )

        st.write(
            profile["description"]
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # Distance information
        # -------------------------------------------------

        st.subheader("📍 Cluster Distance Analysis")

        distance_df = pd.DataFrame({
            "Cluster": [
                f"Cluster {i}"
                for i in range(len(distances))
            ],
            "Distance": distances
        })

        distance_df = distance_df.sort_values(
            "Distance"
        )

        fig = px.bar(
            distance_df,
            x="Cluster",
            y="Distance",
            title="Distance from Each Cluster Centre"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Smaller distance means the rating pattern is closer to that cluster centre."
        )

        # -------------------------------------------------
        # User's strongest preferences
        # -------------------------------------------------

        st.subheader("⭐ Your Highest Ratings")

        user_preferences = pd.Series(input_values)

        top_preferences = (
            user_preferences
            .sort_values(ascending=False)
            .head(5)
        )

        pref_df = pd.DataFrame({
            "Travel Feature": top_preferences.index,
            "Your Rating": top_preferences.values
        })

        fig2 = px.bar(
            pref_df,
            x="Your Rating",
            y="Travel Feature",
            orientation="h",
            title="Your Top Travel Preferences"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# =========================================================
# CLUSTER ANALYSIS
# =========================================================

elif page == "📊 Cluster Analysis":

    st.markdown(
        '<div class="main-title">📊 Cluster Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Visualisation of the traveller groups discovered by K-Means."
    )

    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------

    scaled_data = scaler.transform(df)

    pca = PCA(
        n_components=2
    )

    reduced_data = pca.fit_transform(
        scaled_data
    )

    labels = model.predict(
        scaled_data
    )

    plot_df = pd.DataFrame({

        "PCA 1": reduced_data[:, 0],

        "PCA 2": reduced_data[:, 1],

        "Cluster": labels.astype(str)
    })

    fig = px.scatter(
        plot_df,
        x="PCA 1",
        y="PCA 2",
        color="Cluster",
        title="Traveller Clusters using PCA",
        hover_data=["Cluster"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "PCA is used here only to visualise the high-dimensional rating "
        "data in two dimensions. The actual clustering is performed by K-Means."
    )

    # -----------------------------------------------------
    # Cluster count
    # -----------------------------------------------------

    st.subheader("👥 Number of Tourists in Each Cluster")

    cluster_counts = pd.Series(
        labels
    ).value_counts().sort_index()

    count_df = pd.DataFrame({
        "Cluster": cluster_counts.index.astype(str),
        "Tourists": cluster_counts.values
    })

    fig2 = px.bar(
        count_df,
        x="Cluster",
        y="Tourists",
        title="Tourists per Cluster"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# =========================================================
# DATASET OVERVIEW
# =========================================================

elif page == "📈 Dataset Overview":

    st.markdown(
        '<div class="main-title">📈 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Features",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Clusters",
            model.n_clusters
        )

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("📊 Average Rating by Travel Feature")

    averages = (
        df.mean()
        .sort_values(ascending=False)
    )

    average_df = pd.DataFrame({
        "Feature": averages.index,
        "Average Rating": averages.values
    })

    fig = px.bar(
        average_df,
        x="Average Rating",
        y="Feature",
        orientation="h",
        title="Average Travel Ratings"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )