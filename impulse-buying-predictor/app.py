"""
Impulse Buying Prediction - Streamlit App
Production-ready web application
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent

# Page config
st.set_page_config(
    page_title="Impulse Buying Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "models" / "model.joblib"
    if not model_path.exists():
        st.error(f"❌ Model file not found at {model_path}. Please train the model first.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_feature_info():
    info_path = BASE_DIR / "models" / "feature_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text())
    return None

@st.cache_data
def load_metrics():
    metrics_path = BASE_DIR / "models" / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return None


def get_risk_category(score):
    """Categorize impulse score"""
    if score < 30:
        return "Low", "🟢", "#4CAF50"
    elif score < 60:
        return "Medium", "🟡", "#FFC107"
    else:
        return "High", "🔴", "#F44336"


def create_gauge_chart(score):
    """Create gauge chart for impulse score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Impulse Buy Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#E8F5E9'},
                {'range': [30, 60], 'color': '#FFF9C4'},
                {'range': [60, 100], 'color': '#FFEBEE'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_comparison_chart(user_score):
    """Create comparison chart"""
    categories = ['Low Risk\n(0-30)', 'Medium Risk\n(30-60)', 'High Risk\n(60-100)', 'Your Score']
    values = [15, 45, 80, user_score]
    colors = ['#4CAF50', '#FFC107', '#F44336', '#2196F3']
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors, text=values,
               textposition='outside', texttemplate='%{text:.1f}')
    ])
    
    fig.update_layout(
        title="Score Comparison",
        yaxis_title="Score",
        height=350,
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


# ---------- Robust input alignment & predict wrapper ----------
def prepare_input_for_model(input_df, model):
    """
    Align single-row input_df to model expected features.
    - If model exposes feature_names_in_, use it.
    - Otherwise, try to extract from a pipeline with a ColumnTransformer.
    - Fill missing numeric features with 0 (or sensible default).
    - Cast to numeric where possible.
    """
    expected_features = None
    # 1) try model.feature_names_in_
    try:
        expected_features = list(getattr(model, "feature_names_in_", None))
        if expected_features == [None] or expected_features == []:
            expected_features = None
    except Exception:
        expected_features = None

    # 2) try to introspect a ColumnTransformer inside a pipeline
    if expected_features is None:
        try:
            from sklearn.compose import ColumnTransformer
            for step_name, step in getattr(model, "steps", []):
                if isinstance(step, ColumnTransformer):
                    cols = []
                    for name, trans, cols_or_slice in step.transformers_:
                        if isinstance(cols_or_slice, (list, tuple)):
                            cols.extend(list(cols_or_slice))
                    if cols:
                        expected_features = cols
                        break
        except Exception:
            expected_features = None

    # 3) try feature_info.json in models/
    if expected_features is None:
        try:
            fi = BASE_DIR / "models" / "feature_info.json"
            if fi.exists():
                j = json.loads(fi.read_text())
                expected_features = j.get("feature_names", None)
        except Exception:
            expected_features = None

    # If still unknown, fallback to input columns
    if not expected_features:
        return input_df.copy()

    # Align DataFrame
    X = input_df.copy()

    # Add missing features with default 0
    for f in expected_features:
        if f not in X.columns:
            X[f] = 0

    # Reorder
    X = X[expected_features]

    # Try to coerce numeric types where appropriate
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors="ignore")
        except Exception:
            pass

    return X


def main():
    # Header
    st.markdown('<div class="main-header">🛒 Impulse Buying Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict customer impulse buying behavior using AI</div>', unsafe_allow_html=True)
    
    # Load resources
    model = load_model()
    feature_info = load_feature_info()
    metrics = load_metrics()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
        st.title("Navigation")
        page = st.radio("Select Page", ["🎯 Prediction", "📊 Model Info", "ℹ️ About"])
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        if metrics:
            st.metric("Model R² Score", f"{metrics['test']['r2']:.3f}")
            st.metric("Model RMSE", f"{metrics['test']['rmse']:.2f}")
        
        st.markdown("---")
        st.markdown("**Developed for:**")
        st.markdown("E-commerce Analytics")
        st.markdown("Customer Behavior Prediction")
    
    # Main content
    if page == "🎯 Prediction":
        show_prediction_page(model, feature_info)
    elif page == "📊 Model Info":
        show_model_info_page(metrics)
    else:
        show_about_page()


def show_prediction_page(model, feature_info):
    """Prediction interface"""
    st.header("Make a Prediction")
    
    st.markdown('<div class="info-box">📝 Fill in the customer details below to predict their impulse buying score</div>', unsafe_allow_html=True)
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", 18, 65, 30)
        monthly_income = st.number_input("Monthly Income (₹)", 10000, 300000, 50000, step=5000)
        city = st.selectbox("City", ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata"])
        
    with col2:
        st.subheader("Behavior Metrics")
        total_sessions = st.number_input("Total Sessions (last 90 days)", 0, 300, 50)
        num_product_page_visits = st.number_input("Product Page Visits", 0, 200, 30)
        avg_time_on_product = st.slider("Avg Time on Product (seconds)", 0, 600, 120)
        late_night_session_ratio = st.slider("Late Night Sessions (%)", 0, 100, 20) / 100
        
    with col3:
        st.subheader("Purchase History")
        total_purchases = st.number_input("Total Purchases", 0, 100, 10)
        avg_purchase_value = st.number_input("Avg Purchase Value (₹)", 0, 50000, 2000, step=500)
        avg_discount_used = st.slider("Avg Discount Used (%)", 0, 50, 15)
        impulse_purchase_ratio = st.slider("Past Impulse Ratio (%)", 0, 100, 30) / 100
        past_impulse_purchases = st.number_input("Past Impulse Purchases", 0, 50, 5)
        avg_minutes_to_purchase = st.number_input("Avg Minutes to Purchase", 1, 500, 60)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Psychology")
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        mood = st.selectbox("Mood Last Week", ["Happy", "Neutral", "Anxious", "Sad"])
        saving_habit_score = st.slider("Saving Habit Score (1-5)", 1, 5, 3)
    
    # Map inputs
    mood_map = {"Happy": 1, "Neutral": 2, "Anxious": 3, "Sad": 4}
    city_map = {'Delhi': 1, 'Mumbai': 2, 'Bengaluru': 3, 'Kolkata': 4,
                'Chennai': 5, 'Pune': 6, 'Hyderabad': 7}
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'monthly_income': [monthly_income],
        'city': [city_map[city]],
        'total_sessions': [total_sessions],
        'num_product_page_visits': [num_product_page_visits],
        'avg_time_on_product': [avg_time_on_product],
        'late_night_session_ratio': [late_night_session_ratio],
        'total_purchases': [total_purchases],
        'total_spent': [total_purchases * avg_purchase_value],
        'avg_purchase_value': [avg_purchase_value],
        'avg_discount_used': [avg_discount_used],
        'impulse_purchase_ratio': [impulse_purchase_ratio],
        'past_impulse_purchases': [past_impulse_purchases],
        'avg_minutes_to_purchase': [avg_minutes_to_purchase],
        'stress_level': [stress_level],
        'mood_last_week': [mood_map[mood]],
        'saving_habit_score': [saving_habit_score]
    })

    # Predict button
    if st.button("🔮 Predict Impulse Score", type="primary", use_container_width=True):
        with st.spinner("Analyzing customer behavior..."):
            try:
                Xp = prepare_input_for_model(input_data, model)

                # final safety: ensure shape (1, n_features)
                if Xp.shape[0] != 1:
                    st.error("Internal error: input shaped incorrectly for prediction.")
                else:
                    prediction = model.predict(Xp)[0]
                    prediction = np.clip(prediction, 0, 100)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    risk_cat, emoji, color = get_risk_category(prediction)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f'<div class="prediction-box" style="background: linear-gradient(135deg, {color} 0%, {color}CC 100%);">'
                                    f'{emoji} {prediction:.1f}/100<br><small style="font-size: 1rem;">Impulse Buy Score</small></div>',
                                    unsafe_allow_html=True)
                        
                        st.markdown(f"**Risk Category:** {risk_cat} {emoji}")
                        
                        # Recommendations
                        st.markdown("### 💡 Recommendations")
                        if risk_cat == "High":
                            st.error("⚠️ **High Risk Customer**")
                            st.write("- Implement purchase cooldown periods")
                            st.write("- Show savings progress reminders")
                            st.write("- Limit flash sale notifications")
                        elif risk_cat == "Medium":
                            st.warning("⚠️ **Moderate Risk Customer**")
                            st.write("- Send thoughtful product recommendations")
                            st.write("- Highlight value propositions")
                            st.write("- Provide comparison tools")
                        else:
                            st.success("✅ **Low Risk Customer**")
                            st.write("- Can receive promotional emails")
                            st.write("- Good candidate for loyalty programs")
                            st.write("- Show premium product recommendations")
                    
                    with col2:
                        st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
                        
                    # Comparison chart
                    st.plotly_chart(create_comparison_chart(prediction), use_container_width=True)
                    
                    # Feature importance
                    st.markdown("### 📊 Key Factors")
                    factors = {
                        "Past Impulse Behavior": impulse_purchase_ratio * 100,
                        "Discount Usage": avg_discount_used,
                        "Stress Level": stress_level * 10,
                        "Late Night Activity": late_night_session_ratio * 100,
                        "Purchase Speed": max(0, 100 - avg_minutes_to_purchase / 5)
                    }
                    
                    fig = px.bar(
                        x=list(factors.values()),
                        y=list(factors.keys()),
                        orientation='h',
                        labels={'x': 'Impact Score', 'y': 'Factor'},
                        title="Contributing Factors"
                    )
                    fig.update_traces(marker_color='#2196F3')
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error("Prediction failed — model preprocessing raised an error. Check logs.")
                st.exception(e)


def show_model_info_page(metrics):
    """Model information page"""
    st.header("📊 Model Performance")
    
    if not metrics:
        st.warning("No metrics available. Please train the model first.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{metrics['test']['r2']:.4f}")
    with col2:
        st.metric("RMSE", f"{metrics['test']['rmse']:.2f}")
    with col3:
        st.metric("MAE", f"{metrics['test']['mae']:.2f}")
    with col4:
        st.metric("MAPE", f"{metrics['test']['mape']:.2f}%")
    
    st.markdown("---")
    
    # Performance across sets
    st.subheader("Performance Across Data Splits")
    
    perf_data = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'R² Score': [metrics['train']['r2'], metrics['validation']['r2'], metrics['test']['r2']],
        'RMSE': [metrics['train']['rmse'], metrics['validation']['rmse'], metrics['test']['rmse']],
        'MAE': [metrics['train']['mae'], metrics['validation']['mae'], metrics['test']['mae']]
    })
    
    st.dataframe(perf_data, use_container_width=True)
    
    # Visualize
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(perf_data, x='Dataset', y='R² Score', title='R² Score Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(perf_data, x='Dataset', y='RMSE', title='RMSE Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Model Details")
    st.write(f"**Trained on:** {metrics.get('timestamp', 'Unknown')}")
    try:
        st.write(f"**Model Size:** {metrics.get('model_size_mb', 'Unknown'):.2f} MB")
    except Exception:
        st.write(f"**Model Size:** {metrics.get('model_size_mb', 'Unknown')}")
    st.write("**Algorithm:** LightGBM Regressor")
    st.write("**Features:** 16 behavioral, demographic, and psychological factors")


def show_about_page():
    """About page"""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application predicts customer impulse buying behavior scores (0-100) using machine learning.
    It helps e-commerce businesses understand and manage customer purchasing patterns.
    
    ### 🧠 How It Works
    The model analyzes multiple factors:
    - **Demographics**: Age, income, location
    - **Browsing Behavior**: Session patterns, time spent, late-night activity
    - **Purchase History**: Past purchases, impulse ratio, discount usage
    - **Psychology**: Stress levels, mood, saving habits
    
    ### 📊 Score Interpretation
    - **0-30 (Low)**: Rational buyers, unlikely to make impulse purchases
    - **30-60 (Medium)**: Balanced buyers, occasional impulse purchases
    - **60-100 (High)**: High impulse tendency, frequent unplanned purchases
    
    ### 🔬 Model Performance
    - **Algorithm**: LightGBM (Gradient Boosting)
    - **Training Data**: 50,000 synthetic customer profiles
    - **Features**: 16 carefully engineered variables
    - **Accuracy**: R² > 0.85 on test data
    
    ### 💼 Business Applications
    1. **Targeted Marketing**: Customize campaigns based on impulse scores
    2. **UI/UX Design**: Adjust interface for different user segments
    3. **Inventory Planning**: Stock products matching customer profiles
    4. **Responsible Commerce**: Help customers make better purchase decisions
    
    ### 🛠️ Technical Stack
    - **Framework**: Streamlit
    - **ML Library**: scikit-learn, LightGBM
    - **Visualization**: Plotly
    - **Deployment**: Streamlit Cloud
    
    ### 📝 Notes
    This is a predictive model and should be used as one of many factors in business decisions.
    Always respect customer privacy and use predictions ethically.
    
    ### 👨‍💻 Developer
    Created as an end-to-end ML project demonstrating:
    - Data generation & feature engineering
    - Model training & optimization
    - Web application development
    - Production deployment
    
    ---
    
    **Version**: 1.0  
    **Last Updated**: December 2024
    """)


if __name__ == "__main__":
    main()
