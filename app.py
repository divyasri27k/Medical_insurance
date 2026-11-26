
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MediCost | Insurance Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 55%, #eaf3ff 100%);
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102a43 0%, #173f63 100%);
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background-color: rgba(255,255,255,0.08);
    padding: 12px 14px;
    border-radius: 9px;
    margin-bottom: 7px;
    transition: 0.2s;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: rgba(255,255,255,0.2);
}

/* Main title */
.main-heading {
    color: #153b5b;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sub-heading {
    color: #64748b;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Section headings */
.section-heading {
    color: #173f63;
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #dbeafe;
    box-shadow: 0 3px 12px rgba(30, 64, 110, 0.06);
}

[data-testid="stMetricLabel"] {
    color: #64748b;
    font-weight: 600;
}

/* Form and input containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff;
    border-radius: 14px;
    border: 1px solid #dbeafe;
    padding: 15px;
}

/* Input labels */
.stTextInput label,
.stNumberInput label,
.stSelectbox label {
    color: #24445f;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1769aa, #2586c7);
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #124c80, #1769aa);
    color: white;
    border: none;
    transform: translateY(-2px);
}

/* Prediction result */
.result-card {
    background: linear-gradient(120deg, #e1f5ec, #f0fff7);
    border: 1px solid #a7e5c4;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-top: 20px;
}

.result-label {
    color: #26734d;
    font-size: 17px;
    font-weight: 600;
}

.result-value {
    color: #126b42;
    font-size: 36px;
    font-weight: 800;
}

/* Info cards */
.info-card {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 20px;
    min-height: 135px;
    box-shadow: 0 3px 12px rgba(30, 64, 110, 0.05);
}

.info-title {
    color: #173f63;
    font-size: 18px;
    font-weight: 700;
}

.info-description {
    color: #64748b;
    font-size: 14px;
}

/* Footer */
.footer {
    text-align: center;
    color: #8292a5;
    font-size: 13px;
    padding: 25px 0 5px 0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("Medical_insurance.pkl")


try:
    model = load_model()
    model_loaded = True
    model_error = None
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 0 20px 0;">
        <div style="font-size:48px;">🏥</div>
        <h2 style="margin:0;">MediCost</h2>
        <p style="color:#c7d8e8 !important; font-size:13px;">
            Smart Insurance Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📌 MAIN MENU")

    page = st.radio(
        "Choose a page",
        [
            "🏠 Home",
            "💰 Prediction",
            "📊 Model Insights",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### ⚙️ MODEL STATUS")

    if model_loaded:
        st.success("Model Loaded Successfully")
    else:
        st.error("Model Not Found")

    st.markdown("---")

    st.markdown("### 📁 PROJECT DETAILS")

    st.markdown("""
    **Algorithm:** Linear Regression

    **Problem:** Regression

    **Framework:** Streamlit

    **Language:** Python
    """)

    st.markdown("---")

    st.caption("MediCost | Machine Learning Project")
    st.caption("Developed with Python ❤️")


# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-heading">🏥 Medical Insurance Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-heading">Welcome! Predict estimated medical insurance costs with Machine Learning.</div>',
        unsafe_allow_html=True
    )

    # Dashboard metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🤖 Algorithm", "Linear Regression")

    with c2:
        st.metric("📌 ML Type", "Regression")

    with c3:
        st.metric("📥 Features", "6")

    with c4:
        st.metric(
            "⚡ Model Status",
            "Ready" if model_loaded else "Unavailable"
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">✨ Welcome to MediCost</div>',
        unsafe_allow_html=True
    )

    st.write("""
    MediCost is a Machine Learning application designed to estimate
    medical insurance charges based on personal and lifestyle details.

    Enter your information on the Prediction page to receive an estimated
    insurance cost.
    """)

    st.markdown("### 🔍 What can you do here?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <div style="font-size:30px;">💰</div>
            <div class="info-title">Cost Prediction</div>
            <p class="info-description">
                Estimate medical insurance charges using your details.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div style="font-size:30px;">📊</div>
            <div class="info-title">Model Insights</div>
            <p class="info-description">
                Understand the model and its input features.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <div style="font-size:30px;">🧠</div>
            <div class="info-title">ML Technology</div>
            <p class="info-description">
                Explore how Linear Regression supports cost prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 Go to Insurance Prediction"):
        st.session_state["selected_page"] = "💰 Prediction"
        st.rerun()


# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "💰 Prediction":

    st.markdown(
        '<div class="main-heading">💰 Insurance Cost Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-heading">Enter the required details to estimate your insurance charges.</div>',
        unsafe_allow_html=True
    )

    if not model_loaded:
        st.error("Model could not be loaded.")
        st.code(model_error or "Check Medical_insurance_model.pkl.")
    else:

        st.markdown("### 👤 Personal Information")

        with st.container(border=True):

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input(
                    "🎂 Age",
                    min_value=18,
                    max_value=100,
                    value=25,
                    help="Enter your age in years."
                )

                sex = st.selectbox(
                    "⚧ Gender",
                    ["male", "female"]
                )

                bmi = st.number_input(
                    "⚖️ BMI",
                    min_value=10.0,
                    max_value=60.0,
                    value=25.0,
                    step=0.1,
                    help="Body Mass Index."
                )

            with col2:
                children = st.number_input(
                    "👨‍👩‍👧 Number of Children",
                    min_value=0,
                    max_value=10,
                    value=0
                )

                smoker = st.selectbox(
                    "🚬 Smoking Status",
                    ["no", "yes"]
                )

                region = st.selectbox(
                    "📍 Region",
                    [
                        "southwest",
                        "southeast",
                        "northwest",
                        "northeast"
                    ]
                )

        st.markdown("### 📋 Review Your Details")

        if st.button("🔍 Predict Insurance Charges"):

            input_data = pd.DataFrame([{
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region
            }])

            try:
                prediction = model.predict(input_data)[0]

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">
                        🎉 Estimated Medical Insurance Charges
                    </div>
                    <div class="result-value">
                        ${prediction:,.2f}
                    </div>
                    <p style="color:#47765d;">
                        Your prediction has been generated successfully.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📝 Prediction Summary")

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "Note: This is a machine learning estimate, "
                    "not an official insurance quotation."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ==========================================
# MODEL INSIGHTS PAGE
# ==========================================

elif page == "📊 Model Insights":

    st.markdown(
        '<div class="main-heading">📊 Model Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-heading">Understand the features and technology behind the prediction system.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 Machine Learning Algorithm")

    st.info("""
    Linear Regression is a supervised machine learning algorithm
    used to predict continuous numerical values.

    In this project, it estimates medical insurance charges using
    six input features.
    """)

    st.markdown("### 📌 Features Used")

    feature_data = pd.DataFrame({
        "Feature": [
            "age",
            "sex",
            "bmi",
            "children",
            "smoker",
            "region"
        ],
        "Data Type": [
            "Numerical",
            "Categorical",
            "Numerical",
            "Numerical",
            "Categorical",
            "Categorical"
        ],
        "Description": [
            "Age of the individual",
            "Gender",
            "Body Mass Index",
            "Number of dependent children",
            "Smoking status",
            "Residential region"
        ]
    })

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🔄 Project Workflow")

    st.write("""
    1. Data Collection
    2. Data Cleaning and Preprocessing
    3. Numerical Feature Scaling
    4. Categorical Feature Encoding
    5. Train-Test Split
    6. Linear Regression Model Training
    7. Model Evaluation
    8. Insurance Cost Prediction
    """)


# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-heading">ℹ️ About MediCost</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-heading">Project overview and technologies used.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🎯 Project Objective")

    st.write("""
    The objective of this project is to build a machine learning
    application that estimates individual medical insurance charges
    based on demographic and lifestyle information.
    """)

    st.markdown("### 🛠️ Technologies Used")

    technologies = [
        ("🐍 Python", "Programming language"),
        ("🐼 Pandas", "Data manipulation"),
        ("🔢 NumPy", "Numerical operations"),
        ("🤖 Scikit-learn", "Machine learning"),
        ("🌐 Streamlit", "Web application"),
        ("📦 Joblib", "Model serialization")
    ]

    for technology, description in technologies:
        st.markdown(f"- **{technology}:** {description}")

    st.markdown("### 📌 Project Summary")

    st.write("""
    - Dataset: Medical Insurance Cost dataset
    - ML Category: Supervised Learning
    - Problem Type: Regression
    - Algorithm: Linear Regression
    - Target Variable: charges
    - Input Features: 6
    """)

    st.success("Thank you for exploring MediCost!")


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    © 2026 MediCost | Medical Insurance Cost Prediction
    <br>
    Built using Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)
