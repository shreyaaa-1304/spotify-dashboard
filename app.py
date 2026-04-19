import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
summary_stats = df[["danceability_%","energy_%","valence_%","streams"]].describe()

fig = px.imshow(summary_stats, text_auto=True)
st.plotly_chart(fig, use_container_width=True)

# Set page configuration
st.set_page_config(
    page_title="Spotify Music Data Analysis & Hit Song Prediction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling
def set_dark_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .css-1d391kg {
        background-color: #262730;
    }
    .css-1v0mbdj {
        background-color: #262730;
        color: #FAFAFA;
    }
    .css-1v0mbdj:hover {
        background-color: #31333F;
    }
    .css-17ziqus {
        background-color: #262730;
        color: #FAFAFA;
    }
    .css-1v0mbdj {
        border: 1px solid #4A5568;
    }
    .css-1v0mbdj:focus {
        border-color: #00D4AA;
        box-shadow: 0 0 0 1px #00D4AA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00D4AA;
    }
    .stMetric {
        background-color: #262730;
        border: 1px solid #4A5568;
        padding: 10px;
        border-radius: 10px;
    }
    .stDataFrame {
        background-color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

set_dark_theme()

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('spotify_cleaned.csv')
        
        # Clean and preprocess data
        df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
        df['bpm'] = pd.to_numeric(df['bpm'], errors='coerce')
        
        # Calculate hit songs (top 20% by streams)
        streams_threshold = df['streams'].quantile(0.8)
        df['is_hit'] = df['streams'] >= streams_threshold
        
        # Convert percentage columns to numeric
        percentage_cols = ['danceability_%_valence_%_energy_%_acousticness_%_instrumentalness_%_liveness_%_speechiness_%_']
        if percentage_cols[0] in df.columns:
            # Split the combined column
            split_cols = df[percentage_cols[0]].str.split('_', expand=True)
            df['danceability_%'] = pd.to_numeric(split_cols[0], errors='coerce')
            df['valence_%'] = pd.to_numeric(split_cols[1], errors='coerce') 
            df['energy_%'] = pd.to_numeric(split_cols[2], errors='coerce')
            df['acousticness_%'] = pd.to_numeric(split_cols[3], errors='coerce')
            df['instrumentalness_%'] = pd.to_numeric(split_cols[4], errors='coerce')
            df['liveness_%'] = pd.to_numeric(split_cols[5], errors='coerce')
            df['speechiness_%'] = pd.to_numeric(split_cols[6], errors='coerce')
        else:
            # If columns are already separate
            for col in ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main title
st.title("🎵 Spotify Music Data Analysis & Hit Song Prediction")
st.markdown("---")

# Load data
df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.title("📊 Filters")
    
    # Year filter
    years = sorted(df['released_year'].unique())
    selected_years = st.sidebar.multiselect("Select Years", years, default=years)
    
    # Artist filter
    artists = sorted(df['artist(s)_name'].unique())
    selected_artists = st.sidebar.multiselect("Select Artists", artists, default=[])
    
    # Key filter
    keys = sorted(df['key'].dropna().astype(str).unique())
    selected_keys = st.sidebar.multiselect("Select Keys", keys, default=keys)
    
    # Mode filter
    modes = sorted(df['mode'].unique())
    selected_modes = st.sidebar.multiselect("Select Modes", modes, default=modes)
    
    # Apply filters
    filtered_df = df[
        (df['released_year'].isin(selected_years)) &
        (df['artist(s)_name'].isin(selected_artists) if selected_artists else True) &
        (df['key'].isin(selected_keys)) &
        (df['mode'].isin(selected_modes))
    ]
    
    # Key Metrics Cards
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_streams = filtered_df['streams'].sum()
        st.metric("Total Streams", f"{total_streams/1e9:.2f}B", delta="🎵")
    
    with col2:
        avg_danceability = filtered_df['danceability_%'].mean() * 100 if 'danceability_%' in filtered_df.columns else 0
        st.metric("Danceability %", f"{avg_danceability:.1f}%", delta="💃")
    
    with col3:
        avg_energy = filtered_df['energy_%'].mean() * 100 if 'energy_%' in filtered_df.columns else 0
        st.metric("Energy %", f"{avg_energy:.1f}%", delta="⚡")
    
    with col4:
        avg_valence = filtered_df['valence_%'].mean() * 100 if 'valence_%' in filtered_df.columns else 0
        st.metric("Valence %", f"{avg_valence:.1f}%", delta="😊")
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📦 Box Plot Analysis", "🔮 Linear Regression", "🎯 Clustering", "📈 Detailed Analytics"])
    
    with tab1:
        st.header("📊 Dashboard Overview")
        
        # Hit vs Non-Hit Songs
        col1, col2 = st.columns(2)
        
        with col1:
            hit_counts = filtered_df['is_hit'].value_counts()
            fig_hit = go.Figure(data=[go.Pie(
                labels=['Not Hit', 'Hit'],
                values=[hit_counts.get(False, 0), hit_counts.get(True, 0)],
                hole=0.4,
                marker_colors=['#FF6B6B', '#00D4AA']
            )])
            fig_hit.update_layout(
                title="Distribution of Hit vs Non-Hit Songs",
                font=dict(color='#FAFAFA'),
                paper_bgcolor='#262730',
                plot_bgcolor='#262730'
            )
            st.plotly_chart(fig_hit, use_container_width=True)
        
        with col2:
            # Feature comparison
            feature_comparison = filtered_df.groupby('is_hit')[['danceability_%', 'valence_%', 'energy_%']].mean() * 100
            feature_comparison.index = ['Not Hit', 'Hit']
            
            fig_features = go.Figure()
            for feature in feature_comparison.columns:
                fig_features.add_trace(go.Bar(
                    name=feature.replace('_', ' ').title(),
                    x=feature_comparison.index,
                    y=feature_comparison[feature],
                    marker_color=['#FF6B6B', '#00D4AA'] if feature == 'danceability_%' else ['#4ECDC4', '#45B7D1']
                ))
            
            fig_features.update_layout(
                title="Feature Comparison: Hit vs Non-Hit Songs",
                xaxis_title="Song Type",
                yaxis_title="Average Value (%)",
                font=dict(color='#FAFAFA'),
                paper_bgcolor='#262730',
                plot_bgcolor='#262730',
                barmode='group'
            )
            st.plotly_chart(fig_features, use_container_width=True)
        
        # Streams by Year
        st.subheader("📅 Streams by Released Year")
        streams_by_year = filtered_df.groupby('released_year')['streams'].sum().reset_index()
        streams_by_year = streams_by_year.sort_values('released_year')
        
        fig_year = go.Figure(data=go.Bar(
            x=streams_by_year['released_year'],
            y=streams_by_year['streams'],
            marker_color='#00D4AA'
        ))
        
        fig_year.update_layout(
            title="Total Streams by Release Year",
            xaxis_title="Year",
            yaxis_title="Total Streams",
            font=dict(color='#FAFAFA'),
            paper_bgcolor='#262730',
            plot_bgcolor='#262730'
        )
        st.plotly_chart(fig_year, use_container_width=True)
        
        # Top Artists
        st.subheader("🎤 Popular Artists by Streams")
        top_artists = filtered_df.groupby('artist(s)_name')['streams'].sum().nlargest(10).reset_index()
        
        fig_artists = go.Figure(data=go.Bar(
            x=top_artists['streams'],
            y=top_artists['artist(s)_name'],
            orientation='h',
            marker_color='#45B7D1'
        ))
        
        fig_artists.update_layout(
            title="Top 10 Artists by Total Streams",
            xaxis_title="Total Streams",
            yaxis_title="Artist",
            font=dict(color='#FAFAFA'),
            paper_bgcolor='#262730',
            plot_bgcolor='#262730',
            height=400
        )
        st.plotly_chart(fig_artists, use_container_width=True)
    
    with tab2:
        st.header("📦 Box Plot Analysis (Experiment 1)")
        st.markdown("Distribution of audio features for Hit vs Non-Hit songs")
        
        features_to_plot = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'liveness_%', 'speechiness_%']
        available_features = [f for f in features_to_plot if f in filtered_df.columns]
        
        if available_features:
            selected_feature = st.selectbox("Select Feature for Box Plot", available_features)
            
            fig_box = go.Figure()
            
            for hit_status in [False, True]:
                label = 'Not Hit' if not hit_status else 'Hit'
                color = '#FF6B6B' if not hit_status else '#00D4AA'
                
                fig_box.add_trace(go.Box(
                    y=filtered_df[filtered_df['is_hit'] == hit_status][selected_feature] * 100,
                    name=label,
                    marker_color=color,
                    boxpoints='outliers'
                ))
            
            fig_box.update_layout(
                title=f"Box Plot: {selected_feature.replace('_', ' ').title()} Distribution",
                yaxis_title="Value (%)",
                font=dict(color='#FAFAFA'),
                paper_bgcolor='#262730',
                plot_bgcolor='#262730'
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            summary_stats = filtered_df.groupby('is_hit')[selected_feature].describe() * 100
            summary_stats.index = ['Not Hit', 'Hit']
            st.dataframe(summary_stats.style.background_gradient(cmap='viridis'))
    
    with tab3:
        st.header("🔮 Linear Regression for Hit Prediction (Experiment 2)")
        st.markdown("Predict song popularity using audio features")
        
        # Prepare data for regression
        features = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'liveness_%', 'speechiness_%']
        available_features = [f for f in features if f in filtered_df.columns]
        
        if len(available_features) >= 2:
            X = filtered_df[available_features].fillna(0)
            y = filtered_df['streams'].fillna(0)
            
            # Split data (simple train-test split)
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.2e}")
            with col2:
                st.metric("R² Score", f"{r2:.3f}")
            
            # Feature importance
            st.subheader("📊 Feature Coefficients")
            coefficients = pd.DataFrame({
                'Feature': available_features,
                'Coefficient': model.coef_
            })
            coefficients['Importance'] = np.abs(coefficients['Coefficient'])
            coefficients = coefficients.sort_values('Importance', ascending=False)
            
            fig_coef = go.Figure(data=go.Bar(
                x=coefficients['Feature'],
                y=coefficients['Coefficient'],
                marker_color=['#00D4AA' if c > 0 else '#FF6B6B' for c in coefficients['Coefficient']]
            ))
            
            fig_coef.update_layout(
                title="Feature Coefficients in Linear Regression",
                xaxis_title="Features",
                yaxis_title="Coefficient Value",
                font=dict(color='#FAFAFA'),
                paper_bgcolor='#262730',
                plot_bgcolor='#262730'
            )
            st.plotly_chart(fig_coef, use_container_width=True)
            
            # Prediction vs Actual
            st.subheader("📈 Predictions vs Actual Streams")
            fig_pred = go.Figure()
            
            fig_pred.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='#00D4AA', size=6)
            ))
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='#FF6B6B', dash='dash')
            ))
            
            fig_pred.update_layout(
                title="Predicted vs Actual Streams",
                xaxis_title="Actual Streams",
                yaxis_title="Predicted Streams",
                font=dict(color='#FAFAFA'),
                paper_bgcolor='#262730',
                plot_bgcolor='#262730'
            )
            st.plotly_chart(fig_pred, use_container_width=True)
    
    with tab4:
        st.header("🎯 Clustering Analysis (Experiment 4)")
        st.markdown("Group similar songs based on audio features")
        
        features = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'liveness_%', 'speechiness_%']
        available_features = [f for f in features if f in filtered_df.columns]
        
        if len(available_features) >= 2:
            n_clusters = st.slider("Number of Clusters", 2, 8, 3)
            
            # Prepare data
            X = filtered_df[available_features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to dataframe
            clustered_df = filtered_df.copy()
            clustered_df['cluster'] = cluster_labels
            
            # Cluster visualization
            st.subheader("🎨 Cluster Visualization")
            
            # Use first two features for 2D visualization
            feature_x = st.selectbox("X-axis Feature", available_features, index=0)
            feature_y = st.selectbox("Y-axis Feature", available_features, index=1)
            
            fig_cluster = go.Figure()
            
            colors = ['#00D4AA', '#FF6B6B', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
            
            for i in range(n_clusters):
                cluster_data = clustered_df[clustered_df['cluster'] == i]
                fig_cluster.add_trace(go.Scatter(
                    x=cluster_data[feature_x] * 100,
                    y=cluster_data[feature_y] * 100,
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(color=colors[i % len(colors)], size=6)
                ))
            
            fig_cluster.update_layout(
                title=f"Song Clusters: {feature_x.replace('_', ' ').title()} vs {feature_y.replace('_', ' ').title()}",
                xaxis_title=f"{feature_x.replace('_', ' ').title()} (%)",
                yaxis_title=f"{feature_y.replace('_', ' ').title()} (%)",
                font=dict(color='#FAFAFA'),
                paper_bgcolor='#262730',
                plot_bgcolor='#262730'
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("📊 Cluster Characteristics")
            cluster_stats = clustered_df.groupby('cluster')[available_features].mean() * 100
            cluster_stats['Count'] = clustered_df['cluster'].value_counts().sort_index()
            cluster_stats['Hit_Rate'] = clustered_df.groupby('cluster')['is_hit'].mean() * 100
            
            st.dataframe(cluster_stats.style.background_gradient(cmap='viridis'))
            
            # Cluster hit rate visualization
            fig_hit_rate = go.Figure(data=go.Bar(
                x=[f'Cluster {i}' for i in range(n_clusters)],
                y=clustered_df.groupby('cluster')['is_hit'].mean() * 100,
                marker_color=colors[:n_clusters]
            ))
            
            fig_hit_rate.update_layout(
                title="Hit Rate by Cluster",
                xaxis_title="Cluster",
                yaxis_title="Hit Rate (%)",
                font=dict(color='#FAFAFA'),
                paper_bgcolor='#262730',
                plot_bgcolor='#262730'
            )
            st.plotly_chart(fig_hit_rate, use_container_width=True)
    
    with tab5:
        st.header("📈 Detailed Analytics")
        
        # Top tracks
        st.subheader("🎵 Top Tracks by Streams")
        top_tracks = filtered_df.nlargest(10, 'streams')[['track_name', 'artist(s)_name', 'streams', 'released_year']]
        top_tracks['streams_billion'] = (top_tracks['streams'] / 1e9).round(2)
        st.dataframe(top_tracks[['track_name', 'artist(s)_name', 'streams_billion', 'released_year']].style.background_gradient(cmap='viridis'))
        
        # Audio feature correlations
        st.subheader("🔗 Audio Feature Correlations")
        numeric_features = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'liveness_%', 'speechiness_%', 'streams']
        available_numeric = [f for f in numeric_features if f in filtered_df.columns]
        
        if len(available_numeric) > 1:
            correlation_matrix = filtered_df[available_numeric].corr()
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdYlBu',
                zmid=0,
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig_heatmap.update_layout(
                title="Feature Correlation Matrix",
                font=dict(color='#FAFAFA'),
                paper_bgcolor='#262730',
                plot_bgcolor='#262730'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # BPM Analysis
        st.subheader("🎵 BPM Analysis")
        if 'bpm' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bpm_hist = go.Figure(data=go.Histogram(
                    x=filtered_df['bpm'],
                    nbinsx=20,
                    marker_color='#00D4AA'
                ))
                fig_bpm_hist.update_layout(
                    title="BPM Distribution",
                    xaxis_title="BPM",
                    yaxis_title="Count",
                    font=dict(color='#FAFAFA'),
                    paper_bgcolor='#262730',
                    plot_bgcolor='#262730'
                )
                st.plotly_chart(fig_bpm_hist, use_container_width=True)
            
            with col2:
                bpm_by_hit = filtered_df.groupby('is_hit')['bpm'].mean()
                fig_bpm_hit = go.Figure(data=go.Bar(
                    x=['Not Hit', 'Hit'],
                    y=bpm_by_hit.values,
                    marker_color=['#FF6B6B', '#00D4AA']
                ))
                fig_bpm_hit.update_layout(
                    title="Average BPM: Hit vs Non-Hit",
                    xaxis_title="Song Type",
                    yaxis_title="Average BPM",
                    font=dict(color='#FAFAFA'),
                    paper_bgcolor='#262730',
                    plot_bgcolor='#262730'
                )
                st.plotly_chart(fig_bpm_hit, use_container_width=True)

else:
    st.error("Unable to load the dataset. Please ensure 'spotify_2023.csv' is in the correct directory.")

# Footer
st.markdown("---")
st.markdown("🎵 Spotify Music Data Analysis Dashboard | Created with Streamlit")
st.markdown("📚 Based on Data Analytics Course Experiments")
