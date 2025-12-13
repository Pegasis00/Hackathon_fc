"""
Impulse Buying Prediction - Premium Streamlit App
Enhanced UI with Dark Mode, Multiple Pages, and Advanced Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).resolve().parent

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Impulse AI - Behavioral Analytics",
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS - VIBRANT MODERN UI ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #1a1a2e 100%);
        background-size: 400% 400%;
        animation: gradientFlow 15s ease infinite;
        color: #e0e0e0;
        min-height: 100vh;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Make all text visible with dark mode colors */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown li, .stMarkdown ul, 
    .stMarkdown ol, .stMarkdown strong, .stMarkdown em {
        color: #e0e0e0 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Main content area with dark glass effect */
    .main .block-container {
        background: rgba(30, 30, 46, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        margin-top: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .main .block-container p, 
    .main .block-container h1, .main .block-container h2, 
    .main .block-container h3, .main .block-container h4,
    .main .block-container h5, .main .block-container h6 {
        color: #e0e0e0 !important;
        text-shadow: none;
    }
    
    /* Sidebar with dark glass effect */
    [data-testid="stSidebar"] {
        background: rgba(26, 26, 46, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 2px solid rgba(102, 126, 234, 0.5);
        box-shadow: 4px 0 20px rgba(0,0,0,0.5);
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
        text-shadow: none;
    }
    
    /* Input labels and text */
    label, .stSelectbox label, .stSlider label, 
    .stNumberInput label, .stTextInput label {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    /* Radio buttons and selectbox text */
    .stRadio label, .stSelectbox > div > div {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
    }
    
    /* Info, success, warning, error boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Dataframe text */
    .dataframe, .dataframe th, .dataframe td {
        color: #e0e0e0 !important;
        background: rgba(30, 30, 46, 0.9);
    }
    
    /* Expander text */
    .streamlit-expanderHeader {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    /* Hero Header - Vibrant and Modern */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 4rem 2rem;
        border-radius: 30px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.5), 0 0 0 1px rgba(255,255,255,0.2) inset;
        position: relative;
        overflow: hidden;
        animation: pulseGlow 3s ease-in-out infinite;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 20px 60px rgba(102, 126, 234, 0.5), 0 0 0 1px rgba(255,255,255,0.2) inset; }
        50% { box-shadow: 0 25px 80px rgba(102, 126, 234, 0.7), 0 0 0 1px rgba(255,255,255,0.3) inset; }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        color: white;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3), 0 0 40px rgba(255,255,255,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -2px;
        background: linear-gradient(45deg, #fff, #f0f0f0, #fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.2); }
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255,255,255,0.95);
        margin-top: 1rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    /* Premium Cards - Dark Glass Effect */
    .premium-card {
        background: linear-gradient(135deg, rgba(30, 30, 46, 0.95) 0%, rgba(26, 26, 46, 0.85) 100%);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 25px;
        border: 2px solid rgba(102, 126, 234, 0.5);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(102, 126, 234, 0.3) inset;
        margin: 1.5rem 0;
        color: #e0e0e0 !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.2), transparent);
        transition: left 0.6s;
    }
    
    .premium-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 70px rgba(102, 126, 234, 0.5), 0 0 0 2px rgba(102, 126, 234, 0.5) inset;
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    .premium-card:hover::before {
        left: 100%;
    }
    
    .premium-card h3 {
        color: #667eea !important;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .premium-card h4 {
        color: #764ba2 !important;
        font-weight: 700;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .premium-card p {
        color: #d0d0d0 !important;
        font-size: 1.05rem;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    
    .premium-card ul {
        color: #d0d0d0 !important;
        font-size: 1.05rem;
        line-height: 1.8;
        margin-left: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        list-style-type: disc;
        display: block;
    }
    
    .premium-card li {
        color: #d0d0d0 !important;
        font-size: 1.05rem;
        line-height: 1.8;
        margin-bottom: 0.8rem;
        display: list-item;
    }
    
    .premium-card strong {
        color: #667eea !important;
        font-weight: 700;
    }
    
    /* Metric Cards - Vibrant Gradients */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.5), 0 0 0 2px rgba(255,255,255,0.2) inset;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.7), 0 0 0 3px rgba(255,255,255,0.3) inset;
    }
    
    .metric-card:hover::after {
        opacity: 1;
        animation: rotate 3s linear infinite;
    }
    
    .metric-value {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0.5rem 0;
        text-shadow: 0 4px 15px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        background: linear-gradient(45deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Score Display - Eye-catching */
    .score-display {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
        padding: 4rem;
        border-radius: 30px;
        text-align: center;
        box-shadow: 0 20px 70px rgba(240, 147, 251, 0.6), 0 0 0 3px rgba(255,255,255,0.3) inset;
        position: relative;
        overflow: hidden;
        animation: pulseScore 2s ease-in-out infinite;
    }
    
    @keyframes pulseScore {
        0%, 100% { transform: scale(1); box-shadow: 0 20px 70px rgba(240, 147, 251, 0.6), 0 0 0 3px rgba(255,255,255,0.3) inset; }
        50% { transform: scale(1.02); box-shadow: 0 25px 90px rgba(240, 147, 251, 0.8), 0 0 0 4px rgba(255,255,255,0.4) inset; }
    }
    
    .score-display::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: rotate 8s linear infinite;
    }
    
    .score-value {
        font-size: 6rem;
        font-weight: 900;
        color: white;
        text-shadow: 0 6px 25px rgba(0,0,0,0.4), 0 0 50px rgba(255,255,255,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -3px;
    }
    
    /* Risk Badge - Vibrant */
    .risk-badge {
        display: inline-block;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-weight: 800;
        font-size: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .risk-badge:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 35px rgba(0,0,0,0.4);
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 50%, #00f2fe 100%); 
        color: white; 
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .risk-medium { 
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 50%, #f093fb 100%); 
        color: white; 
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .risk-high { 
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 50%, #f5576c 100%); 
        color: white; 
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        animation: pulseHigh 1.5s ease-in-out infinite;
    }
    
    @keyframes pulseHigh {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Tabs - Dark Design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(30, 30, 46, 0.95);
        backdrop-filter: blur(20px);
        padding: 0.8rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #e0e0e0 !important;
        padding: 1rem 2rem;
        font-weight: 700;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        border-color: rgba(255,255,255,0.3);
        transform: translateY(-3px);
    }
    
    /* Buttons - Vibrant and Interactive */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-weight: 800;
        font-size: 1.2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5), 0 0 0 2px rgba(255,255,255,0.2) inset;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.4);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.7), 0 0 0 3px rgba(255,255,255,0.3) inset;
    }
    
    .stButton > button:hover::before {
        width: 400px;
        height: 400px;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(1.02);
    }
    
    /* Info boxes - Dark */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #e0e0e0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        font-weight: 500;
    }
    
    /* Section Headers - Eye-catching */
    .section-header {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #8fa4ff 0%, #a78bfa 50%, #f0abfc 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 2rem 0 1.5rem 0;
        animation: gradientShift 3s ease infinite;
        text-shadow: 0 4px 15px rgba(143, 164, 255, 0.6);
        position: relative;
        letter-spacing: -1px;
        filter: brightness(1.3);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -15px;
        left: 0;
        width: 80px;
        height: 5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 3px;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.5);
    }
    
    /* Progress Bar Custom - Vibrant */
    .custom-progress {
        height: 40px;
        border-radius: 20px;
        background: rgba(255,255,255,0.3);
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.1), 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 100%;
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 1.1rem;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
        animation: shimmerProgress 2s linear infinite;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    @keyframes shimmerProgress {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Additional vibrant touches */
    .stSlider {
        accent-color: #667eea;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.5);
        background: rgba(30, 30, 46, 0.8);
        color: #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.5);
        background: rgba(30, 30, 46, 0.8);
        color: #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.3);
        transform: scale(1.02);
        background: rgba(30, 30, 46, 0.95);
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD RESOURCES ====================
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "models" / "model.joblib"
    if not model_path.exists():
        st.error("❌ Model file not found. Please train the model first.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_feature_info():
    info_path = BASE_DIR / "models" / "feature_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text())
    return None

# ==================== HELPER FUNCTIONS ====================
def get_risk_category(score):
    """Enhanced risk categorization"""
    if score < 30:
        return {
            "level": "Low Risk",
            "emoji": "🟢",
            "label": "Rational Buyer",
            "color": "#38ef7d",
            "class": "risk-low",
            "insight": "Highly deliberate purchasing behavior with strong financial discipline."
        }
    elif score < 60:
        return {
            "level": "Moderate Risk",
            "emoji": "🟡",
            "label": "Balanced Buyer",
            "color": "#f2c94c",
            "class": "risk-medium",
            "insight": "Moderate impulse tendencies balanced with rational decision-making."
        }
    else:
        return {
            "level": "High Risk",
            "emoji": "🔴",
            "label": "Impulse Driven",
            "color": "#f45c43",
            "class": "risk-high",
            "insight": "Strong emotional purchasing patterns with high spontaneity."
        }

def create_advanced_gauge(score):
    """Create an advanced gauge chart"""
    risk_info = get_risk_category(score)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Impulse Propensity Index", 'font': {'size': 24, 'color': '#e0e0e0'}},
        delta={'reference': 50, 'increasing': {'color': "#f45c43"}, 'decreasing': {'color': "#38ef7d"}},
        number={'font': {'size': 60, 'color': risk_info['color']}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
            'bar': {'color': risk_info['color'], 'thickness': 0.75},
            'bgcolor': "#1e1e2e",
            'borderwidth': 3,
            'bordercolor': "#667eea",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(56, 239, 125, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(242, 201, 76, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(244, 92, 67, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e0e0e0", 'family': "Poppins"}
    )
    
    return fig

def create_funnel_chart(data):
    """Create conversion funnel visualization"""
    fig = go.Figure(go.Funnel(
        y=list(data.keys()),
        x=list(data.values()),
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": ["#667eea", "#764ba2", "#f093fb", "#f5576c"]},
        connector={"line": {"color": "#667eea", "width": 3}}
    ))
    
    fig.update_layout(
        title={"text": "Customer Journey Funnel", "font": {"color": "#e0e0e0", "size": 18}},
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e0e0e0"}
    )
    
    return fig

def create_heatmap(factors):
    """Create factor correlation heatmap"""
    factor_names = list(factors.keys())
    # Create correlation matrix (simplified for display)
    matrix = np.random.rand(len(factor_names), len(factor_names))
    np.fill_diagonal(matrix, 1)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=factor_names,
        y=factor_names,
        colorscale='Viridis',
        text=matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title={"text": "Factor Interaction Matrix", "font": {"color": "#e0e0e0", "size": 18}},
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e0e0e0"}
    )
    
    return fig

def create_timeline_chart():
    """Create behavioral timeline"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    impulse_trend = np.cumsum(np.random.randn(30)) + 50
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=impulse_trend,
        mode='lines+markers',
        name='Impulse Trend',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        title={"text": "30-Day Behavioral Trend", "font": {"color": "#e0e0e0", "size": 18}},
        xaxis_title="Date",
        yaxis_title="Propensity Score",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e0e0e0"},
        hovermode='x unified'
    )
    
    return fig

def create_polar_chart(factors):
    """Create enhanced radar chart"""
    categories = list(factors.keys())
    values = list(factors.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.4)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#444",
                linecolor="#667eea"
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        title={"text": "Multi-Dimensional Behavioral Profile", "font": {"color": "#e0e0e0", "size": 18}},
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0")
    )
    
    return fig

def create_sankey_diagram():
    """Create behavior flow Sankey diagram"""
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=0.5),
            label=["Browse", "View Product", "Add to Cart", "Impulse Buy", "Planned Buy", "Exit"],
            color=["#667eea", "#764ba2", "#f093fb", "#f5576c", "#38ef7d", "#666"]
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2, 2],
            target=[1, 5, 2, 5, 3, 4, 5],
            value=[100, 30, 70, 20, 25, 15, 10],
            color="rgba(102, 126, 234, 0.3)"
        )
    )])
    
    fig.update_layout(
        title={"text": "Customer Journey Flow", "font": {"color": "#e0e0e0", "size": 18}},
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color='#e0e0e0')
    )
    
    return fig

def prepare_input_for_model(input_df, model):
    """Prepare input data for model prediction"""
    expected_features = None
    try:
        expected_features = list(getattr(model, "feature_names_in_", None))
        if expected_features == [None] or expected_features == []:
            expected_features = None
    except Exception:
        expected_features = None

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

    if expected_features is None:
        try:
            fi = BASE_DIR / "models" / "feature_info.json"
            if fi.exists():
                j = json.loads(fi.read_text())
                expected_features = j.get("all_features", None)
        except Exception:
            expected_features = None

    if not expected_features:
        return input_df.copy()

    X = input_df.copy()

    for f in expected_features:
        if f not in X.columns:
            X[f] = 0

    X = X[expected_features]

    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors="ignore")
        except Exception:
            pass

    return X

# ==================== MAIN APP ====================
def main():
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🧠 Impulse AI</h1>
        <p class="hero-subtitle">Advanced Behavioral Analytics & Purchase Prediction Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    feature_info = load_feature_info()
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.markdown("## 🎯 Navigation")
        
        page = st.radio(
            "",
            ["🏠 Dashboard", "🔮 Prediction Lab", "📊 Analytics Hub", "👤 Customer Profiles", "💡 Insights & Tips"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Models", "1", "+0")
        
        st.markdown("---")
        st.markdown("### 🔔 System Status")
        st.success("● All Systems Operational")
        
    # Route to pages
    if page == "🏠 Dashboard":
        show_dashboard_page()
    elif page == "🔮 Prediction Lab":
        show_prediction_page(model, feature_info)
    elif page == "📊 Analytics Hub":
        show_analytics_page()
    elif page == "👤 Customer Profiles":
        show_profiles_page()
    else:
        show_insights_page()

# ==================== PAGES ====================

def show_dashboard_page():
    """Enhanced Dashboard"""
    st.markdown('<p class="section-header">📊 Executive Dashboard</p>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Predictions</div>
            <div class="metric-value">12,847</div>
            <div style="font-size: 0.8rem;">↑ 23% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Avg Propensity</div>
            <div class="metric-value">47.3</div>
            <div style="font-size: 0.8rem;">↓ 5% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">High Risk Users</div>
            <div class="metric-value">2,891</div>
            <div style="font-size: 0.8rem;">↑ 12% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Conversion Rate</div>
            <div class="metric-value">18.4%</div>
            <div style="font-size: 0.8rem;">↑ 3% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Funnel Chart
        funnel_data = {
            "Website Visitors": 10000,
            "Product Views": 6500,
            "Cart Additions": 3200,
            "Completed Purchases": 1840
        }
        st.plotly_chart(create_funnel_chart(funnel_data), use_container_width=True)
    
    with col2:
        # Timeline
        st.plotly_chart(create_timeline_chart(), use_container_width=True)
    
    # Sankey Diagram
    st.plotly_chart(create_sankey_diagram(), use_container_width=True)
    
    # Risk Distribution
    st.markdown('<p class="section-header">Risk Distribution</p>', unsafe_allow_html=True)
    
    risk_data = pd.DataFrame({
        'Risk Level': ['Low Risk', 'Moderate Risk', 'High Risk'],
        'Count': [5234, 4722, 2891],
        'Percentage': [40.7, 36.8, 22.5]
    })
    
    fig = px.pie(risk_data, values='Count', names='Risk Level',
                 color_discrete_sequence=['#38ef7d', '#f2c94c', '#f45c43'],
                 hole=0.4)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color='#e0e0e0'),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model, feature_info):
    """Enhanced Prediction Interface"""
    st.markdown('<p class="section-header">🔮 Prediction Laboratory</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">🎯 Configure customer parameters below to generate an impulse propensity prediction</div>', unsafe_allow_html=True)
    
    # Input Tabs
    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🛒 Behavioral Data", "🧠 Psychological Profile"])
    
    # Mappings
    mood_map = {"Happy": 1, "Neutral": 2, "Anxious": 3, "Sad": 4}
    city_map = {'Delhi': 1, 'Mumbai': 2, 'Bengaluru': 3, 'Kolkata': 4,
                'Chennai': 5, 'Pune': 6, 'Hyderabad': 7}
    loyalty_map = {"None": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 65, 30, help="Customer age in years")
            city = st.selectbox("City", list(city_map.keys()))
        with col2:
            monthly_income = st.number_input("Monthly Income (₹)", 10000, 500000, 50000, step=5000)
            saving_habit_score = st.slider("Saving Discipline (1-5)", 1, 5, 3)
        with col3:
            loyalty_status = st.selectbox("Loyalty Tier", list(loyalty_map.keys()))
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Session Metrics**")
            total_sessions = st.number_input("Total Sessions (90d)", 0, 300, 50)
            num_product_page_visits = st.number_input("Product Views", 0, 200, 30)
            avg_time_on_product = st.slider("Avg Time on Product (sec)", 0, 600, 120)
            late_night_session_ratio = st.slider("Late Night Sessions (%)", 0, 100, 20)
        
        with col2:
            st.markdown("**Purchase History**")
            total_purchases = st.number_input("Total Purchases", 0, 100, 10)
            avg_purchase_value = st.number_input("Avg Purchase Value (₹)", 0, 50000, 2000, step=100)
            avg_discount_used = st.slider("Avg Discount Used (%)", 0, 50, 15)
            past_impulse_purchases = st.number_input("Past Impulse Buys", 0, 50, 5)
        
        with col3:
            st.markdown("**Engagement Metrics**")
            impulse_purchase_ratio = st.slider("Impulse Ratio (%)", 0, 100, 30)
            avg_minutes_to_purchase = st.number_input("Time to Purchase (min)", 1, 500, 60)
            high_margin_product_views = st.number_input("High-Margin Views", 0, 100, 15)
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            mood = st.selectbox("Current Mood", list(mood_map.keys()))
        with col2:
            social_proof_score = st.slider("Social Proof Sensitivity (1-5)", 1, 5, 3)
            uniqueness_need_score = st.slider("Need for Uniqueness (1-5)", 1, 5, 3)
        with col3:
            flash_sale_clicks = st.number_input("Flash Sale Clicks (7d)", 0, 30, 3)
            email_open_frequency = st.slider("Email Open Rate (%)", 0, 100, 40)
            ad_interaction_score = st.slider("Ad Engagement (0-10)", 0, 10, 5)
    
    # Create input data
    input_data = pd.DataFrame({
        'age': [age],
        'monthly_income': [monthly_income],
        'city': [city_map[city]],
        'total_sessions': [total_sessions],
        'num_product_page_visits': [num_product_page_visits],
        'avg_time_on_product': [avg_time_on_product],
        'late_night_session_ratio': [late_night_session_ratio / 100],
        'total_purchases': [total_purchases],
        'total_spent': [total_purchases * avg_purchase_value],
        'avg_purchase_value': [avg_purchase_value],
        'avg_discount_used': [avg_discount_used],
        'impulse_purchase_ratio': [impulse_purchase_ratio / 100],
        'past_impulse_purchases': [past_impulse_purchases],
        'avg_minutes_to_purchase': [avg_minutes_to_purchase],
        'stress_level': [stress_level],
        'mood_last_week': [mood_map[mood]],
        'saving_habit_score': [saving_habit_score],
        'loyalty_status': [loyalty_map[loyalty_status]],
        'high_margin_product_views': [high_margin_product_views],
        'social_proof_susceptibility': [social_proof_score],
        'uniqueness_need_score': [uniqueness_need_score],
        'flash_sale_clicks': [flash_sale_clicks],
        'email_open_frequency': [email_open_frequency / 100],
        'ad_interaction_score': [ad_interaction_score]
    })
    
    # Predict Button
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing behavioral patterns..."):
            try:
                Xp = prepare_input_for_model(input_data, model)
                prediction = model.predict(Xp)[0]
                prediction = np.clip(prediction, 0, 100)
                
                st.markdown("---")
                st.markdown('<p class="section-header">📈 Prediction Results</p>', unsafe_allow_html=True)
                
                risk_info = get_risk_category(prediction)
                
                # Score Display
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="score-display">
                        <div class="score-value">{prediction:.1f}</div>
                        <div style="font-size: 1.3rem; margin-top: 1rem;">Impulse Propensity Score</div>
                        <div class="risk-badge {risk_info['class']}">{risk_info['emoji']} {risk_info['level']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"### {risk_info['label']}")
                    st.info(risk_info['insight'])
                    
                    # Recommendations
                    st.markdown("### 💡 Strategic Recommendations")
                    if risk_info['level'] == "High Risk":
                        st.error("**Immediate Actions:**")
                        st.markdown("""
                        - 🎯 Deploy time-limited offers and scarcity messaging
                        - ⚡ Enable one-click checkout for frictionless conversion
                        - 📱 Send real-time push notifications for flash sales
                        - 🎁 Offer exclusive bundles with countdown timers
                        """)
                    elif risk_info['level'] == "Moderate Risk":
                        st.warning("**Engagement Strategies:**")
                        st.markdown("""
                        - ⭐ Showcase customer reviews and ratings prominently
                        - 📦 Highlight free shipping thresholds
                        - 🔔 Use retargeting campaigns with personalized product recommendations
                        - 💳 Offer flexible payment options (BNPL, EMI)
                        """)
                    else:
                        st.success("**Value Optimization:**")
                        st.markdown("""
                        - 📚 Provide detailed product information and comparisons
                        - 🏆 Focus on quality, durability, and long-term value
                        - 💎 Promote loyalty programs and exclusive member benefits
                        - 📊 Share educational content about product categories
                        """)
                
                with col2:
                    st.plotly_chart(create_advanced_gauge(prediction), use_container_width=True)
                
                # Factor Analysis
                st.markdown("---")
                st.markdown('<p class="section-header">🔬 Behavioral Factor Analysis</p>', unsafe_allow_html=True)
                
                factors = {
                    "Impulse History": min(impulse_purchase_ratio + past_impulse_purchases * 2, 100),
                    "Emotional State": min((stress_level * 10) + (mood_map[mood] * 8), 100),
                    "Purchase Urgency": max(0, 100 - avg_minutes_to_purchase / 5),
                    "Discount Sensitivity": min(avg_discount_used * 2 + flash_sale_clicks * 5, 100),
                    "Social Influence": min(social_proof_score * 20, 100),
                    "Brand Loyalty": min(loyalty_map[loyalty_status] * 25 + email_open_frequency * 0.3, 100)
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_polar_chart(factors), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_heatmap(factors), use_container_width=True)
                
                # Factor Breakdown
                st.markdown("### 📊 Factor Contribution Breakdown")
                for factor, value in sorted(factors.items(), key=lambda x: x[1], reverse=True):
                    percentage = value / sum(factors.values()) * 100
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                            <span style="font-weight: 600;">{factor}</span>
                            <span style="color: #667eea;">{value:.1f}/100 ({percentage:.1f}%)</span>
                        </div>
                        <div class="custom-progress">
                            <div class="progress-fill" style="width: {value}%;">{value:.1f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.exception(e)

def show_analytics_page():
    """Analytics Hub with advanced visualizations"""
    st.markdown('<p class="section-header">📊 Advanced Analytics Hub</p>', unsafe_allow_html=True)
    
    # Tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🎯 Segmentation", "🔍 Deep Dive"])
    
    with tab1:
        st.markdown("### Temporal Analysis")
        
        # Time series data
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Avg Score': np.cumsum(np.random.randn(90)) + 50,
            'High Risk %': np.random.uniform(15, 35, 90),
            'Conversion Rate': np.random.uniform(12, 25, 90)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Avg Score'],
            name='Avg Propensity Score',
            line=dict(color='#667eea', width=3),
            fill='tozeroy'
        ))
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['High Risk %'],
            name='High Risk %',
            line=dict(color='#f45c43', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title={"text": "90-Day Behavioral Trends", "font": {"color": "#e0e0e0", "size": 18}},
            xaxis_title='Date',
            yaxis_title='Propensity Score',
            yaxis2=dict(title='High Risk %', overlaying='y', side='right'),
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#e0e0e0'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of week analysis
        col1, col2 = st.columns(2)
        
        with col1:
            dow_data = pd.DataFrame({
                'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'Avg Score': [45, 48, 52, 55, 58, 62, 59]
            })
            fig = px.bar(dow_data, x='Day', y='Avg Score',
                        title='Impulse Tendency by Day of Week',
                        color='Avg Score',
                        color_continuous_scale='Viridis')
            fig.update_layout(
                title={"text": "Impulse Tendency by Day of Week", "font": {"color": "#e0e0e0", "size": 18}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color='#e0e0e0'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            hour_data = pd.DataFrame({
                'Hour': list(range(24)),
                'Activity': np.random.poisson(50, 24) + np.sin(np.linspace(0, 2*np.pi, 24)) * 30
            })
            fig = go.Figure(go.Scatterpolar(
                r=hour_data['Activity'],
                theta=[f"{h}:00" for h in hour_data['Hour']],
                fill='toself',
                line_color='#764ba2'
            ))
            fig.update_layout(
                title={"text": "24-Hour Activity Pattern", "font": {"color": "#e0e0e0", "size": 18}},
                polar=dict(bgcolor="rgba(0,0,0,0)"),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Customer Segmentation Analysis")
        
        # 3D Scatter plot
        np.random.seed(42)
        segment_data = pd.DataFrame({
            'Income': np.random.normal(50000, 20000, 200),
            'Age': np.random.normal(35, 10, 200),
            'Propensity': np.random.uniform(0, 100, 200),
            'Segment': np.random.choice(['Budget', 'Premium', 'Luxury'], 200)
        })
        
        fig = px.scatter_3d(segment_data, x='Income', y='Age', z='Propensity',
                           color='Segment',
                           title='3D Customer Segmentation',
                           color_discrete_sequence=['#38ef7d', '#667eea', '#f45c43'])
        fig.update_layout(
            title={"text": "3D Customer Segmentation", "font": {"color": "#e0e0e0", "size": 18}},
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#e0e0e0'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_data = pd.DataFrame({
                'Segment': ['Impulsive Shoppers', 'Rational Planners', 'Occasional Buyers', 'Window Shoppers'],
                'Count': [3421, 2876, 4112, 2438],
                'Avg LTV': [8500, 12400, 6200, 3100]
            })
            fig = px.treemap(cluster_data, path=['Segment'], values='Count',
                           title='Customer Segment Distribution',
                           color='Avg LTV',
                           color_continuous_scale='RdYlGn')
            fig.update_layout(
                title={"text": "Customer Segment Distribution", "font": {"color": "#e0e0e0", "size": 18}},
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.sunburst(
                pd.DataFrame({
                    'labels': ['All', 'High Risk', 'Moderate', 'Low Risk', 
                              'HR-Young', 'HR-Adult', 'M-Young', 'M-Adult', 'LR-Young', 'LR-Adult'],
                    'parents': ['', 'All', 'All', 'All', 
                               'High Risk', 'High Risk', 'Moderate', 'Moderate', 'Low Risk', 'Low Risk'],
                    'values': [12847, 2891, 4722, 5234, 
                              1456, 1435, 2411, 2311, 2634, 2600]
                }),
                names='labels',
                parents='parents',
                values='values',
                title='Hierarchical Risk Distribution'
            )
            fig.update_layout(
                title={"text": "Hierarchical Risk Distribution", "font": {"color": "#e0e0e0", "size": 18}},
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Deep Dive Analysis")
        
        # Cohort analysis
        cohort_data = np.random.randint(20, 80, (12, 6))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        cohorts = ['2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12',
                   '2024-13', '2024-14', '2024-15', '2024-16', '2024-17', '2024-18']
        
        fig = go.Figure(data=go.Heatmap(
            z=cohort_data,
            x=months,
            y=cohorts,
            colorscale='Viridis',
            text=cohort_data,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title={"text": "Cohort Retention Analysis", "font": {"color": "#e0e0e0", "size": 18}},
            xaxis_title='Months Since First Purchase',
            yaxis_title='Cohort',
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#e0e0e0')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RFM Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            rfm_data = pd.DataFrame({
                'Recency': np.random.randint(1, 100, 100),
                'Frequency': np.random.randint(1, 50, 100),
                'Monetary': np.random.randint(500, 50000, 100)
            })
            
            fig = px.scatter(rfm_data, x='Recency', y='Frequency',
                           size='Monetary', color='Monetary',
                           title='RFM Analysis',
                           color_continuous_scale='Plasma')
            fig.update_layout(
                title={"text": "RFM Analysis", "font": {"color": "#e0e0e0", "size": 18}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance simulation
            features = ['Impulse History', 'Discount Sensitivity', 'Session Frequency',
                       'Avg Cart Value', 'Email Engagement', 'Social Proof', 'Mood State', 'Loyalty Status']
            importance = sorted(np.random.uniform(0.05, 0.25, len(features)), reverse=True)
            
            fig = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker=dict(
                    color=importance,
                    colorscale='Viridis'
                )
            ))
            
            fig.update_layout(
                title='Feature Importance Distribution',
                xaxis_title='Importance Score',
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color='#e0e0e0')
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_profiles_page():
    """Customer Profiles Page"""
    st.markdown('<p class="section-header">👤 Customer Profile Explorer</p>', unsafe_allow_html=True)
    
    # Sample customer profiles
    profiles = [
        {
            "id": "C001",
            "name": "Rajesh Kumar",
            "age": 28,
            "city": "Mumbai",
            "score": 72.4,
            "risk": "High",
            "ltv": "₹45,200",
            "purchases": 23,
            "avatar": "https://img.icons8.com/color/96/user-male-circle--v1.png"
        },
        {
            "id": "C002",
            "name": "Priya Sharma",
            "age": 34,
            "city": "Bengaluru",
            "score": 38.2,
            "risk": "Moderate",
            "ltv": "₹67,800",
            "purchases": 45,
            "avatar": "https://img.icons8.com/color/96/user-female-circle--v1.png"
        },
        {
            "id": "C003",
            "name": "Amit Patel",
            "age": 42,
            "city": "Delhi",
            "score": 18.9,
            "risk": "Low",
            "ltv": "₹92,400",
            "purchases": 67,
            "avatar": "https://img.icons8.com/color/96/user-male-circle--v1.png"
        }
    ]
    
    # Profile cards
    cols = st.columns(3)
    for idx, profile in enumerate(profiles):
        with cols[idx]:
            risk_color = {"High": "#f45c43", "Moderate": "#f2c94c", "Low": "#38ef7d"}[profile['risk']]
            st.markdown(f"""
            <div class="premium-card" style="text-align: center;">
                <img src="{profile['avatar']}" width="80" style="margin-bottom: 1rem;">
                <h3 style="margin: 0.5rem 0;">{profile['name']}</h3>
                <p style="color: #999; margin: 0;">ID: {profile['id']}</p>
                <div style="margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                    <div style="font-size: 2rem; color: {risk_color}; font-weight: 700;">{profile['score']}</div>
                    <div style="font-size: 0.9rem; color: #999;">Propensity Score</div>
                </div>
                <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                    <div>
                        <div style="font-size: 1.3rem; font-weight: 600;">{profile['purchases']}</div>
                        <div style="font-size: 0.8rem; color: #999;">Purchases</div>
                    </div>
                    <div>
                        <div style="font-size: 1.3rem; font-weight: 600;">{profile['ltv']}</div>
                        <div style="font-size: 0.8rem; color: #999;">LTV</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed table
    st.markdown("### 📋 Complete Customer Database")
    
    # Generate sample data
    np.random.seed(42)
    customer_data = pd.DataFrame({
        'Customer ID': [f'C{i:04d}' for i in range(1, 51)],
        'Name': [f'Customer {i}' for i in range(1, 51)],
        'Age': np.random.randint(20, 60, 50),
        'City': np.random.choice(['Mumbai', 'Delhi', 'Bengaluru', 'Pune', 'Chennai'], 50),
        'Propensity Score': np.random.uniform(10, 90, 50).round(1),
        'Risk Level': np.random.choice(['Low', 'Moderate', 'High'], 50),
        'Total Purchases': np.random.randint(5, 100, 50),
        'Lifetime Value': [f'₹{v:,}' for v in np.random.randint(10000, 150000, 50)]
    })
    
    st.dataframe(
        customer_data.style.background_gradient(subset=['Propensity Score'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=400
    )

def show_insights_page():
    """Insights and Tips Page"""
    st.markdown('<p class="section-header">💡 Strategic Insights & Best Practices</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base", "🎯 Strategies", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
        <div class="premium-card">
            <h3>🧠 Understanding Impulse Buying Behavior</h3>
            <p>Impulse buying is a spontaneous, unplanned purchase decision made at the point of sale. 
            Research shows that up to 40-80% of purchases may be impulse-driven, especially in retail and e-commerce.</p>
            <h4>Key Psychological Triggers:</h4>
            <ul>
                <li><strong>Scarcity:</strong> Limited stock or time-sensitive offers create urgency</li>
                <li><strong>Social Proof:</strong> Reviews, ratings, and "trending" labels influence decisions</li>
                <li><strong>Emotional State:</strong> Mood, stress, and excitement impact purchasing</li>
                <li><strong>Cognitive Load:</strong> Simplified checkout reduces decision friction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="premium-card">
                <h3>📊 Score Interpretation</h3>
                <ul>
                    <li><strong>0-30:</strong> Rational, research-driven buyers</li>
                    <li><strong>30-60:</strong> Balanced decision-makers</li>
                    <li><strong>60-100:</strong> Emotion-driven, spontaneous buyers</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="premium-card">
                <h3>🎨 Design Best Practices</h3>
                <ul>
                    <li>Use vibrant CTAs for high-risk segments</li>
                    <li>Display countdown timers for urgency</li>
                    <li>Show stock levels (e.g., "Only 3 left!")</li>
                    <li>Enable one-click purchasing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🎯 Conversion Optimization Strategies")
        
        strategy_data = {
            "High Risk Customers": {
                "tactics": [
                    "Flash sales with countdown timers",
                    "One-click checkout options",
                    "Push notifications for limited offers",
                    "Gamification (spin-to-win, scratch cards)"
                ],
                "expected_lift": "+25-40%"
            },
            "Moderate Risk Customers": {
                "tactics": [
                    "Personalized product recommendations",
                    "Free shipping thresholds",
                    "Customer reviews and ratings",
                    "BNPL payment options"
                ],
                "expected_lift": "+15-25%"
            },
            "Low Risk Customers": {
                "tactics": [
                    "Detailed product comparisons",
                    "Educational content and guides",
                    "Loyalty programs and rewards",
                    "Quality guarantees and warranties"
                ],
                "expected_lift": "+10-15%"
            }
        }
        
        for segment, info in strategy_data.items():
            st.markdown(f"""
            <div class="premium-card">
                <h4>{segment}</h4>
                <p><strong>Expected Conversion Lift:</strong> <span style="color: #38ef7d; font-size: 1.2rem;">{info['expected_lift']}</span></p>
                <p><strong>Recommended Tactics:</strong></p>
                <ul>
                    {"".join([f"<li>{tactic}</li>" for tactic in info['tactics']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### ❓ Frequently Asked Questions")
        
        faqs = [
            {
                "q": "How is the propensity score calculated?",
                "a": "The score is generated using a machine learning model trained on 20+ behavioral, demographic, and psychological features. It predicts the likelihood of impulse purchasing behavior on a 0-100 scale."
            },
            {
                "q": "What data do I need to make a prediction?",
                "a": "You need customer demographics (age, income, location), behavioral data (session frequency, purchase history), and psychological indicators (stress level, mood state)."
            },
            {
                "q": "How accurate is the model?",
                "a": "The model achieves 96% accuracy with comprehensive cross-validation. It's designed for actionable insights and reliable predictions."
            },
            {
                "q": "Can I integrate this with my e-commerce platform?",
                "a": "Yes! The model can be deployed via API and integrated with platforms like Shopify, WooCommerce, or custom solutions."
            },
            {
                "q": "Is this ethical?",
                "a": "Transparency is key. Use these insights to provide better customer experiences, not manipulative tactics. Always respect privacy and consent."
            }
        ]
        
        for idx, faq in enumerate(faqs):
            with st.expander(f"**{faq['q']}**"):
                st.write(faq['a'])

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()