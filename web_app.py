#!/usr/bin/env python3
"""
Web Interface for Autonomous Machine Learning Agent

This module provides a Streamlit web interface for the autonomous ML agent.
"""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.orchestrator import AutonomousMLAgent, PipelineConfig
from src.utils.llm_client import LLMClient, LLMConfig, MockLLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🤖 Autonomous ML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """Main web application"""

    # Header
    st.markdown(
        '<h1 class="main-header">🤖 Autonomous Machine Learning Agent</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            An intelligent, LLM-orchestrated machine learning pipeline that automatically ingests tabular datasets,
            cleans and preprocesses data, trains multiple models, and optimizes them for target metrics.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Dataset (CSV)",
            type=["csv"],
            help="Upload your tabular dataset in CSV format",
        )

        # Target column selection
        target_column = st.text_input(
            "🎯 Target Column",
            placeholder="Enter target column name",
            help="Name of the column you want to predict",
        )

        # Optimization metric
        optimization_metric = st.selectbox(
            "📊 Optimization Metric",
            options=["accuracy", "precision", "recall", "f1", "auc"],
            index=0,
            help="Metric to optimize during model training",
        )

        # Time budget
        time_budget = st.slider(
            "⏱️ Time Budget (minutes)",
            min_value=5,
            max_value=120,
            value=60,
            help="Maximum time to spend on the entire pipeline",
        )

        # Model settings
        st.subheader("🤖 Model Settings")
        max_models = st.slider(
            "Maximum Models",
            min_value=3,
            max_value=15,
            value=10,
            help="Maximum number of models to train",
        )

        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of cross-validation folds",
        )

        # Feature flags
        st.subheader("🔧 Features")
        enable_ensemble = st.checkbox("Enable Ensemble", value=True)
        enable_interpretability = st.checkbox("Enable Interpretability", value=True)
        enable_meta_learning = st.checkbox("Enable Meta-learning", value=True)

        # LLM settings
        st.subheader("🧠 LLM Settings")
        use_mock_llm = st.checkbox("Use Mock LLM (Testing)", value=False)

        if not use_mock_llm:
            llm_provider = st.selectbox(
                "LLM Provider", options=["openai", "anthropic"], index=0
            )
            llm_model = st.text_input(
                "LLM Model",
                value=(
                    "gpt-4" if llm_provider == "openai" else "claude-3-sonnet-20240229"
                ),
            )

        # Run button
        run_button = st.button(
            "🚀 Start Autonomous Pipeline", type="primary", use_container_width=True
        )

    # Main content area
    if uploaded_file is not None:
        # Load and display dataset
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Dataset Preview")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Dataset Info:**")
                st.write(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                st.write(
                    f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                )
                st.write(f"- Missing values: {df.isnull().sum().sum()}")

            with col2:
                st.write("**Columns:**")
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    missing = df[col].isnull().sum()
                    st.write(f"- {col}: {dtype} ({missing} missing)")

            # Display first few rows
            st.write("**First 5 rows:**")
            st.dataframe(df.head(), use_container_width=True)

            # Target column validation
            if target_column:
                if target_column not in df.columns:
                    st.error(
                        f"❌ Target column '{target_column}' not found in dataset!"
                    )
                    return

                st.success(f"✅ Target column '{target_column}' found!")

                # Show target distribution
                st.subheader("🎯 Target Distribution")
                target_counts = df[target_column].value_counts()

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Value Counts:**")
                    st.dataframe(
                        target_counts.reset_index().rename(
                            columns={"index": "Value", target_column: "Count"}
                        )
                    )

                with col2:
                    # Create pie chart
                    fig = px.pie(
                        values=target_counts.values,
                        names=target_counts.index,
                        title=f"Distribution of {target_column}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return

    # Run pipeline
    if run_button and uploaded_file is not None and target_column:
        with st.spinner("🤖 Running autonomous ML pipeline..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    dataset_path = tmp_file.name

                try:
                    # Create configuration
                    config = PipelineConfig(
                        dataset_path=dataset_path,
                        target_column=target_column,
                        optimization_metric=optimization_metric,
                        time_budget=time_budget * 60,  # Convert to seconds
                        max_models=max_models,
                        cross_validation_folds=cv_folds,
                        enable_ensemble=enable_ensemble,
                        enable_interpretability=enable_interpretability,
                        enable_meta_learning=enable_meta_learning,
                    )

                    # Initialize LLM client
                    if use_mock_llm:
                        MockLLMClient()
                    else:
                        llm_config = LLMConfig(provider=llm_provider, model=llm_model)
                        LLMClient(llm_config)

                    # Create and run agent
                    agent = AutonomousMLAgent(
                        dataset_path=dataset_path,
                        target_column=target_column,
                        **vars(config),
                    )

                    # Run pipeline
                    results = asyncio.run(agent.run())

                    # Display results
                    display_results(results)

                finally:
                    # Clean up temporary file
                    os.unlink(dataset_path)

            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
                logger.error(f"Pipeline failed: {e}")

    elif run_button:
        if uploaded_file is None:
            st.error("❌ Please upload a dataset first!")
        if not target_column:
            st.error("❌ Please specify a target column!")


def display_results(results):
    """Display pipeline results"""
    st.success("🎉 Pipeline completed successfully!")

    # Results overview
    st.subheader("📊 Results Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Best Model", results.best_model.__class__.__name__)

    with col2:
        best_score = results.leaderboard.iloc[0][results.leaderboard.columns[1]]
        st.metric("Best Score", f"{best_score:.4f}")

    with col3:
        st.metric("Training Time", f"{results.training_time:.1f}s")

    with col4:
        st.metric("Models Trained", results.total_iterations)

    # Leaderboard
    st.subheader("🏆 Model Leaderboard")

    # Create leaderboard visualization
    leaderboard_df = results.leaderboard.copy()

    # Create bar chart for model comparison
    fig = go.Figure()

    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric in leaderboard_df.columns:
            fig.add_trace(
                go.Bar(
                    name=metric.capitalize(),
                    x=leaderboard_df["model_name"],
                    y=leaderboard_df[metric],
                    text=[f"{val:.3f}" for val in leaderboard_df[metric]],
                    textposition="auto",
                )
            )

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode="group",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed leaderboard table
    st.dataframe(leaderboard_df, use_container_width=True)

    # Feature importance
    if results.feature_importance:
        st.subheader("🔍 Feature Importance")

        # Create feature importance visualization
        feature_importance_df = pd.DataFrame(
            list(results.feature_importance.items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)

        fig = px.bar(
            feature_importance_df.tail(15),  # Top 15 features
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Most Important Features",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature importance table
        st.dataframe(feature_importance_df, use_container_width=True)

    # Model insights
    st.subheader("💡 AI-Generated Insights")

    st.markdown(
        f"""
    <div class="metric-card">
        <p style="font-size: 1.1rem; line-height: 1.6;">
            {results.model_insights}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Download results
    st.subheader("💾 Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Download leaderboard
        csv = results.leaderboard.to_csv(index=False)
        st.download_button(
            label="📊 Download Leaderboard",
            data=csv,
            file_name="leaderboard.csv",
            mime="text/csv",
        )

    with col2:
        # Download feature importance
        import json

        feature_importance_json = json.dumps(results.feature_importance, indent=2)
        st.download_button(
            label="🔍 Download Feature Importance",
            data=feature_importance_json,
            file_name="feature_importance.json",
            mime="application/json",
        )

    with col3:
        # Download insights
        st.download_button(
            label="💡 Download Insights",
            data=results.model_insights,
            file_name="insights.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
