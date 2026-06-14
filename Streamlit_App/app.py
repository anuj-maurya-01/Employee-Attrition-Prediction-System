import streamlit as st
import os
import pandas as pd
import numpy as np
from utils import load_ml_assets, predict_attrition, extract_risk_factors

# Page Configuration
st.set_page_config(
    page_title="Executive People Analytics - Attrition Prediction System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load ML Assets
assets = load_ml_assets()

# Theme Selector in Sidebar (Setup first so we can apply styles)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image("https://img.icons8.com/nolan/96/combo-chart.png", width=60)
st.sidebar.markdown("## PEOPLE INSIGHTS")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎨 Theme Customization")
theme_mode = st.sidebar.selectbox(
    "Select Interface Mode", 
    ["Light Mode", "Dark Mode"], 
    index=0,
    help="Toggle between Light and Dark visual theme modes."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 Executive Toolset")
st.sidebar.info("This system translates complex organizational telemetry into predictive retention strategies.")

if assets['status'] == 'error':
    st.sidebar.error("⚠️ Model assets missing!")
    st.error(f"Failed to initialize preprocessor. Error: {assets.get('error_msg')}")
    st.sidebar.markdown("Run `python train_models.py` in the workspace to serialize model binaries.")
else:
    st.sidebar.success("✅ Prediction engines online.")
st.sidebar.markdown("---")
st.sidebar.markdown("🏛️ **Enterprise Dashboard v1.3**")
st.sidebar.markdown("© **anuj_enterprises 2026**")


# Apply Dynamic Theme CSS variables based on selection
if theme_mode == "Light Mode":
    st.markdown("""
    <style>
        /* Light Theme CSS Variables */
        :root {
            --text-color: #0f172a;
            --sub-text-color: #334155;
            --kpi-title-color: #475569;
            --border-color: rgba(99, 102, 241, 0.25);
            --card-bg: #ffffff;
            --card-hover-border: #4f46e5;
            --card-shadow: 0 4px 20px 0 rgba(148, 163, 184, 0.15);
            --card-hover-shadow: 0 10px 30px 0 rgba(99, 102, 241, 0.2);
            --kpi-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            --bg-gradient: radial-gradient(circle at top right, #f8fafc 0%, #f1f5f9 60%, #e2e8f0 100%);
            --th-bg: rgba(99, 102, 241, 0.15);
            --th-color: #1e1b4b;
            --hr-gradient: linear-gradient(to right, rgba(99, 102, 241, 0.4), rgba(99, 102, 241, 0.05));
            --form-bg: #ffffff;
            
            /* Sidebar variables for Light Mode */
            --sidebar-bg: #f8fafc;
            --sidebar-border: rgba(99, 102, 241, 0.25);
            --sidebar-text: #0f172a;
            --sidebar-subtext: #475569;
        }
        
        /* Specific Targeted Selectors for Light Mode */
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: #0f172a !important;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #1e1b4b !important;
        }
        
        /* Widget labels (sliders, selects, inputs) */
        .stWidget label, [data-testid="stWidgetLabel"] p {
            color: #0f172a !important;
            font-weight: 600 !important;
        }
        
        /* Radio option text */
        div[data-testid="stRadio"] label p, div[data-testid="stRadio"] span {
            color: #0f172a !important;
            font-weight: 500 !important;
        }
        
        /* Tabs text styling */
        button[data-baseweb="tab"] p {
            color: #475569 !important;
            font-weight: 600 !important;
        }
        button[aria-selected="true"] p {
            color: #4f46e5 !important;
            font-weight: 700 !important;
        }
        
        .stApp {
            background: var(--bg-gradient) !important;
        }
        
        .premium-title {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        
        .premium-subtitle {
            color: #334155 !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        /* Dark Theme CSS Variables */
        :root {
            --text-color: #f1f5f9;
            --sub-text-color: #94a3b8;
            --kpi-title-color: #94a3b8;
            --border-color: rgba(99, 102, 241, 0.18);
            --card-bg: rgba(15, 23, 42, 0.7);
            --card-hover-border: #818cf8;
            --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            --card-hover-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.18);
            --kpi-gradient: linear-gradient(135deg, #a5b4fc 0%, #c084fc 100%);
            --bg-gradient: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 50%, #020617 100%);
            --th-bg: rgba(99, 102, 241, 0.1);
            --th-color: #a5b4fc;
            --hr-gradient: linear-gradient(to right, rgba(99, 102, 241, 0.3), rgba(99, 102, 241, 0.05));
            --form-bg: rgba(15, 23, 42, 0.45);
            
            /* Sidebar variables for Dark Mode */
            --sidebar-bg: #030712;
            --sidebar-border: rgba(99, 102, 241, 0.15);
            --sidebar-text: #cbd5e1;
            --sidebar-subtext: #64748b;
        }
        
        /* Specific Targeted Selectors for Dark Mode */
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: #e2e8f0 !important;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #818cf8 !important;
        }
        
        /* Widget labels (sliders, selects, inputs) */
        .stWidget label, [data-testid="stWidgetLabel"] p {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
        }
        
        /* Radio option text */
        div[data-testid="stRadio"] label p, div[data-testid="stRadio"] span {
            color: #cbd5e1 !important;
            font-weight: 500 !important;
        }
        
        /* Tabs text styling */
        button[data-baseweb="tab"] p {
            color: #94a3b8 !important;
            font-weight: 600 !important;
        }
        button[aria-selected="true"] p {
            color: #a5b4fc !important;
            font-weight: 700 !important;
        }
        
        .stApp {
            background: var(--bg-gradient) !important;
        }
        
        .premium-title {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        
        .premium-subtitle {
            color: #94a3b8 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Common Styles using CSS Variables (Safe targeted overrides)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Sidebar styling overrides */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
    }
    
    /* Target all texts in Sidebar to respect sidebar-text and sidebar-subtext */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: var(--sidebar-text) !important;
    }
    
    [data-testid="stSidebar"] .stInfo p {
        color: var(--sidebar-text) !important;
    }

    /* Premium Typography */
    .premium-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    
    .premium-subtitle {
        font-size: 16px;
        margin-bottom: 24px;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #6366f1 !important;
        margin-top: 28px;
        margin-bottom: 14px;
        border-left: 4px solid #6366f1;
        padding-left: 12px;
        letter-spacing: -0.2px;
    }

    /* Glassmorphism Card Style */
    .premium-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: var(--card-shadow) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-4px);
        border-color: var(--card-hover-border) !important;
        box-shadow: var(--card-hover-shadow) !important;
    }
    
    /* Card content specific scoping */
    .premium-card p, .premium-card li, .premium-card div, .premium-card span {
        color: var(--text-color) !important;
    }
    
    /* Styled metric text inside cards */
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: var(--kpi-title-color) !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 44px;
        font-weight: 800;
        background: var(--kpi-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        line-height: 1;
        margin: 6px 0;
    }
    
    .metric-desc {
        font-size: 13px;
        color: var(--sub-text-color) !important;
    }

    /* High Risk Result Card (Safe text scopes) */
    .risk-alert-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.03) 100%) !important;
        border: 1.5px solid rgba(239, 68, 68, 0.6) !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px -5px rgba(239, 68, 68, 0.2);
        backdrop-filter: blur(12px);
        margin-bottom: 24px;
    }
    .risk-alert-high div, .risk-alert-high p, .risk-alert-high span {
        color: #b91c1c !important;
        font-weight: 600 !important;
    }
    
    /* Low Risk Result Card (Safe text scopes) */
    .risk-alert-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.03) 100%) !important;
        border: 1.5px solid rgba(16, 185, 129, 0.6) !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px -5px rgba(16, 185, 129, 0.2);
        backdrop-filter: blur(12px);
        margin-bottom: 24px;
    }
    .risk-alert-low div, .risk-alert-low p, .risk-alert-low span {
        color: #047857 !important;
        font-weight: 600 !important;
    }
    
    /* Image caption styling */
    div[data-testid="caption"] {
        color: var(--sub-text-color) !important;
        font-weight: 600 !important;
        margin-top: 6px;
        font-size: 14px !important;
    }
    
    /* Table Headers Styling */
    th {
        background-color: var(--th-bg) !important;
        color: var(--th-color) !important;
        font-weight: 600 !important;
    }
    
    /* Custom HR Line */
    .premium-hr {
        border: 0;
        height: 1px;
        background: var(--hr-gradient) !important;
        margin: 24px 0;
    }
    
    /* Form background and borders override */
    form[data-testid="stForm"] {
        background: var(--form-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 24px !important;
    }
    
    /* Make tab font look nice */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# Header Section
st.markdown('<div class="premium-title">Employee Attrition Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="premium-subtitle">Predictive retention analytics and workforce risk forecasting using machine learning classification models.</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_overview, tab_eda, tab_predict, tab_compare = st.tabs([
    "🏠 Executive Summary", 
    "📊 People Analytics Dashboard", 
    "🔮 Retention Risk Predictor", 
    "🏆 Model Benchmark Suite"
])

# ----------------- Tab 1: Executive Summary -----------------
with tab_overview:
    st.markdown('<div class="section-header">Project Overview & Business Value</div>', unsafe_allow_html=True)
    st.markdown("""
    Employee attrition represents a significant capital leak for organizations. Beyond recruitment costs, the loss of institutional knowledge, operational drag, and impact on team morale degrade organizational performance. 
    
    This platform introduces a proactive **Machine Learning lifecycle solution** to identify retention risks. By leveraging 30 employee features across demographics, compensation, satisfaction, and longevity metrics, our models predict leaving risk and identify contributing factors.
    """)
    
    # KPI Metric Cards Row
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.markdown("""
        <div class="premium-card">
            <div class="metric-title">Cohort Size</div>
            <div class="metric-value">1,470</div>
            <div class="metric-desc">Total employee records analyzed in dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        st.markdown("""
        <div class="premium-card">
            <div class="metric-title">Feature Dimensions</div>
            <div class="metric-value">30</div>
            <div class="metric-desc">Workplace & demographic variables modeled</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        st.markdown("""
        <div class="premium-card">
            <div class="metric-title">Historical Baseline</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #f87171 0%, #f43f5e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">16.1%</div>
            <div class="metric-desc">Average historical attrition rate (imbalanced data)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Analytical Methodology</div>', unsafe_allow_html=True)
    st.markdown("""
    The system processes raw data and performs predictions through a multi-stage machine learning pipeline:
    1. **Data Preprocessing**: Checks for duplicates, drop constant columns (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`), and applies **IQR Capping** to treat outliers in compensation.
    2. **Data Balancing**: Employs **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set to prevent model bias towards the majority class (active employees).
    3. **Ensemble & Linear Classifiers**: Trains and cross-evaluates **Logistic Regression**, **Random Forest**, and **XGBoost** engines.
    """)

# ----------------- Tab 2: People Analytics Dashboard -----------------
with tab_eda:
    st.markdown('<div class="section-header">Exploratory Insights & Feature Importances</div>', unsafe_allow_html=True)
    st.markdown("Analyzing key factors that drive employee attrition. The graphs show feature importance values extracted from tree models:")
    
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("#### 🌲 Random Forest Feature Importances")
        rf_imp_path = os.path.join(assets_dir, "rf_feature_importances.png")
        if os.path.exists(rf_imp_path):
            st.image(rf_imp_path, use_container_width=True)
        else:
            st.info("Feature importance plot missing. Run model training.")
            
    with col_e2:
        st.markdown("#### ⚡ XGBoost Feature Importances")
        xgb_imp_path = os.path.join(assets_dir, "xgb_feature_importances.png")
        if os.path.exists(xgb_imp_path):
            st.image(xgb_imp_path, use_container_width=True)
        else:
            st.info("XGBoost importance plot missing.")
            
    st.markdown('<div class="premium-hr"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Key Bivariate Analytical Findings</div>', unsafe_allow_html=True)
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        st.markdown("""
        <div class="premium-card">
            <div class="card-title" style="color: #ef4444; font-size: 16px; font-weight: bold;">🔥 Overtime Burnout</div>
            <div style="font-size: 14px; margin-top: 10px; line-height: 1.6;">
                Employees working <strong>Overtime</strong> exhibit an attrition rate of nearly <strong>30%</strong>, compared to just 10% for standard hours. This represents a primary driver of voluntary departures.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d2:
        st.markdown("""
        <div class="premium-card">
            <div class="card-title" style="color: #6366f1; font-size: 16px; font-weight: bold;">💰 Compensation Floor</div>
            <div style="font-size: 14px; margin-top: 10px; line-height: 1.6;">
                Attrition is heavily concentrated in the lower income brackets. Roles such as <strong>Sales Representatives</strong> and <strong>Laboratory Technicians</strong> earning under <strong>$3,000/month</strong> show the highest turnover.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d3:
        st.markdown("""
        <div class="premium-card">
            <div class="card-title" style="color: #10b981; font-size: 16px; font-weight: bold;">⏱️ Early Tenure Volatility</div>
            <div style="font-size: 14px; margin-top: 10px; line-height: 1.6;">
                The first <strong>2 years</strong> of tenure are the most critical. Attrition rates are highest for newer hires, suggesting the need for improved onboarding support and early career pathing.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- Tab 3: Retention Risk Predictor -----------------
with tab_predict:
    st.markdown('<div class="section-header">Predictive Simulation</div>', unsafe_allow_html=True)
    st.markdown("Input employee telemetry below. The predictor will evaluate their attrition risk profile in real-time.")
    
    if assets['status'] == 'error':
        st.warning("Please run model training to initialize predictive models.")
    else:
        # Professional grouped form layout using layout containers
        with st.form("professional_predictor_form"):
            
            # Sub-section 1: Personal & Demographic
            st.markdown("##### 👤 1. Personal & Education Profile")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                age = st.slider("Age (Years)", 18, 65, 34, help="Select employee age.")
                gender = st.selectbox("Gender", ["Female", "Male"])
            with col_p2:
                marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
                education = st.selectbox("Education Level", [1, 2, 3, 4, 5], format_func=lambda x: {
                    1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"
                }[x])
            with col_p3:
                edu_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
            
            st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
            
            # Sub-section 2: Role & Department
            st.markdown("##### 💼 2. Job Role & Workplace Status")
            col_j1, col_j2, col_j3 = st.columns(3)
            with col_j1:
                dept = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
                # Dynamic role mapping based on department
                roles_map = {
                    "Research & Development": ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Research Director", "Manager"],
                    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
                    "Human Resources": ["Human Resources", "Manager"]
                }
                job_role = st.selectbox("Job Role", roles_map[dept])
            with col_j2:
                job_level = st.slider("Job Level (1 - 5)", 1, 5, 2)
                job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2, format_func=lambda x: {
                    1: "Low", 2: "Medium", 3: "High", 4: "Very High"
                }[x])
            with col_j3:
                perf_rating = st.selectbox("Performance Rating", [3, 4], format_func=lambda x: {
                    3: "Excellent", 4: "Outstanding"
                }[x])
                
            st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
                
            # Sub-section 3: Compensation & Tenure
            st.markdown("##### 💰 3. Compensation & Tenure History")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=6200, step=100)
                pct_hike = st.slider("Percent Salary Hike (%)", 11, 25, 15)
                stock_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=1)
            with col_c2:
                num_companies = st.slider("Num Companies Worked", 0, 9, 2)
                total_work_years = st.slider("Total Working Years", 0, 40, 10)
                years_at_company = st.slider("Years At Company", 0, 40, 6)
            with col_c3:
                years_in_role = st.slider("Years In Current Role", 0, 20, 4)
                years_promo = st.slider("Years Since Last Promotion", 0, 15, 2)
                years_manager = st.slider("Years With Current Manager", 0, 20, 4)
                
                # Hidden/low impact numeric rates required by pipeline
                daily_rate = 800
                hourly_rate = 65
                monthly_rate = 14000
                
            st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
                
            # Sub-section 4: Satisfaction & Workplace Life
            st.markdown("##### 🌟 4. Satisfaction & Work-Life Balance")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                travel = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
                distance = st.slider("Commute Distance (Miles)", 1, 30, 8)
            with col_s2:
                overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
                training = st.slider("Training Times Last Year", 0, 6, 2)
            with col_s3:
                job_sat = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2, format_func=lambda x: {
                    1: "Low", 2: "Medium", 3: "High", 4: "Very High"
                }[x])
                env_sat = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2, format_func=lambda x: {
                    1: "Low", 2: "Medium", 3: "High", 4: "Very High"
                }[x])
            
            # Satisfaction & Work-life row 2
            col_s4, col_s5 = st.columns(2)
            with col_s4:
                rel_sat = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2, format_func=lambda x: {
                    1: "Low", 2: "Medium", 3: "High", 4: "Very High"
                }[x])
            with col_s5:
                work_life = st.selectbox("Work-Life Balance Rating", [1, 2, 3, 4], index=2, format_func=lambda x: {
                    1: "Bad", 2: "Good", 3: "Better", 4: "Best"
                }[x])
                
            st.markdown('<div class="premium-hr"></div>', unsafe_allow_html=True)
            
            # Model Selection & Submission
            col_submit, col_select = st.columns([1, 2])
            with col_select:
                selected_model_name = st.radio(
                    "Classifier Engine", 
                    ["XGBoost", "Random Forest", "Logistic Regression"],
                    horizontal=True,
                    help="Choose the algorithm to run the prediction."
                )
            with col_submit:
                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                submit_btn = st.form_submit_button("Assess Retention Risk", use_container_width=True)
                
        # Handle form submission
        if submit_btn:
            # Construct feature dict matching training columns
            input_dict = {
                'Age': age,
                'BusinessTravel': travel,
                'DailyRate': daily_rate,
                'Department': dept,
                'DistanceFromHome': distance,
                'Education': education,
                'EducationField': edu_field,
                'EnvironmentSatisfaction': env_sat,
                'Gender': gender,
                'HourlyRate': hourly_rate,
                'JobInvolvement': job_involvement,
                'JobLevel': job_level,
                'JobRole': job_role,
                'JobSatisfaction': job_sat,
                'MaritalStatus': marital,
                'MonthlyIncome': monthly_income,
                'MonthlyRate': monthly_rate,
                'NumCompaniesWorked': num_companies,
                'OverTime': overtime,
                'PercentSalaryHike': pct_hike,
                'PerformanceRating': perf_rating,
                'RelationshipSatisfaction': rel_sat,
                'StockOptionLevel': stock_level,
                'TotalWorkingYears': total_work_years,
                'TrainingTimesLastYear': training,
                'WorkLifeBalance': work_life,
                'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': years_in_role,
                'YearsSinceLastPromotion': years_promo,
                'YearsWithCurrManager': years_manager
            }
            
            # Predict
            model = assets['models'][selected_model_name]
            preprocessor = assets['preprocessor']
            original_cols = assets['original_cols']
            feature_names = assets['feature_names']
            
            pred, prob = predict_attrition(model, preprocessor, original_cols, input_dict)
            
            st.markdown('<div class="section-header">Simulation Result Assessment</div>', unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if pred == 1:
                    st.markdown(f"""
                    <div class="risk-alert-high">
                        <div style="font-size: 22px; font-weight: 800; color: #b91c1c; margin-bottom: 8px;">🚨 HIGH RISK</div>
                        <div style="font-size: 32px; font-weight: 800; color: #991b1b; margin-bottom: 8px;">{prob*100:.1f}%</div>
                        <div style="font-size: 14px; color: #7f1d1d; line-height: 1.5; font-weight: 600;">
                            Probability of departure exceeds the safety threshold. Action recommended: Review workload, scheduling, and compensation.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-alert-low">
                        <div style="font-size: 22px; font-weight: 800; color: #047857; margin-bottom: 8px;">✅ LOW RISK</div>
                        <div style="font-size: 32px; font-weight: 800; color: #065f46; margin-bottom: 8px;">{prob*100:.1f}%</div>
                        <div style="font-size: 14px; color: #064e3b; line-height: 1.5; font-weight: 600;">
                            Employee exhibits profile metrics aligned with organizational retention. Maintain standard feedback loops.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_res2:
                st.markdown("#### Probability Gauge")
                st.progress(prob)
                
                # Extract and highlight risk factors
                risk_factors = extract_risk_factors(model, preprocessor, original_cols, input_dict, feature_names)
                if risk_factors:
                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                    st.markdown("##### Identified Risk Triggers:")
                    for r in risk_factors:
                        st.markdown(f"⚠️ **{r['feature']}** represents a risk factor for this employee profile.")
                else:
                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                    st.success("No critical risk triggers found. Key satisfaction, overtime, and compensation indicators are in the safe zone.")

# ----------------- Tab 4: Model Benchmark Suite -----------------
with tab_compare:
    st.markdown('<div class="section-header">Classifier Benchmarks & Metrics</div>', unsafe_allow_html=True)
    st.markdown("Comparative testing metrics across the test dataset (20% holdout split):")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(os.path.dirname(current_dir), "Model")
    metrics_path = os.path.join(model_dir, "metrics_comparison.csv")
    
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path, index_col=0)
        
        # Display styled table
        st.dataframe(df_metrics.style.format("{:.4f}").highlight_max(color='#1e3a8a', axis=0), use_container_width=True)
        
        st.markdown('<div class="premium-hr"></div>', unsafe_allow_html=True)
        
        col_c_plot1, col_c_plot2 = st.columns(2)
        
        with col_c_plot1:
            st.markdown("#### Receiver Operating Characteristic (ROC) Comparison")
            roc_curve_path = os.path.join(assets_dir, "roc_comparison.png")
            if os.path.exists(roc_curve_path):
                st.image(roc_curve_path, use_container_width=True)
                
        with col_c_plot2:
            st.markdown("#### Model Performance Metrics Breakdown")
            comp_chart_path = os.path.join(assets_dir, "metrics_comparison.png")
            if os.path.exists(comp_chart_path):
                st.image(comp_chart_path, use_container_width=True)
                
        st.markdown('<div class="premium-hr"></div>', unsafe_allow_html=True)
        
        st.markdown("#### Confusion Matrices")
        col_cm1, col_cm2, col_cm3 = st.columns(3)
        with col_cm1:
            cm1_path = os.path.join(assets_dir, "logistic_regression_cm.png")
            if os.path.exists(cm1_path):
                st.image(cm1_path, caption="Logistic Regression Confusion Matrix", use_container_width=True)
        with col_cm2:
            cm2_path = os.path.join(assets_dir, "random_forest_cm.png")
            if os.path.exists(cm2_path):
                st.image(cm2_path, caption="Random Forest Confusion Matrix", use_container_width=True)
        with col_cm3:
            cm3_path = os.path.join(assets_dir, "xgboost_cm.png")
            if os.path.exists(cm3_path):
                st.image(cm3_path, caption="XGBoost Confusion Matrix", use_container_width=True)
                
    else:
        st.info("Performance comparative metrics missing. Run `train_models.py` first.")

# Centered Page Footer
st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
st.markdown("<hr style='border: 0; height: 1px; background: var(--hr-gradient); margin: 24px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: var(--sub-text-color); font-size: 13px; font-weight: 500; padding-bottom: 24px;'>© anuj_enterprises 2026. All rights reserved.</div>", unsafe_allow_html=True)
