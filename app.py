
from pathlib import Path
from datetime import date

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------
# 1. PROJECT SETTINGS
# ------------------------------------------------

PROJECT_FOLDER = Path(__file__).parent

MODEL_FILE = (
    PROJECT_FOLDER / "models" / "shopsense_model.joblib"
)

st.set_page_config(
    page_title="ShopSense",
    page_icon="🛍️",
    layout="wide"
)


# ------------------------------------------------
# 2. DASHBOARD DESIGN
# ------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fb;
    color: #17243a;
}

[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #17243a;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Main headings */

h1, h2, h3, h4 {
    color: #17243a !important;
}

/* Text */

.stApp p, .stApp label {
    color: #475569;
}

/* Hero section */

.hero {
    background: linear-gradient(
        120deg,
        #17243a,
        #285078
    );

    padding: 30px;
    border-radius: 16px;
    margin-bottom: 24px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 36px;
}

.hero p {
    color: #e2eaf5 !important;
    font-size: 16px;
}

/* Cards */

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5ebf2;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"] {
    color: #17243a !important;
}

/* Form */

div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e5ebf2;
    border-radius: 14px;
    padding: 20px;
}

/* Input boxes */

[data-baseweb="input"] input,
[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #17243a !important;
}

/* Button */

div[data-testid="stFormSubmitButton"] button {
    background: #087f70 !important;
    color: white !important;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 45px;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# 3. LOAD TRAINED MODEL
# ------------------------------------------------

if not MODEL_FILE.exists():

    st.error(
        "Trained model not found. "
        "Run python train_model.py first."
    )

    st.stop()


model = joblib.load(MODEL_FILE)


# ------------------------------------------------
# 4. SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.title("🛍️ ShopSense")

    st.caption("CUSTOMER PURCHASE INTELLIGENCE")

    page = st.radio(
        "Navigation",
        ["Home", "About ShopSense"]
    )

    st.divider()

    st.write("**Algorithm:** Random Forest")

    st.write("**Project:** Machine Learning")

    st.write("**Interface:** Streamlit")


# ------------------------------------------------
# 5. MAIN HEADER
# ------------------------------------------------

st.markdown("""
<div class="hero">

    <h1>🛍️ ShopSense</h1>

    <p>
    Customer Shopping Session &
    Purchase Intention Predictor
    </p>

    <p>
    Understand shopping behaviour and estimate
    whether a visitor may make a purchase.
    </p>

</div>
""", unsafe_allow_html=True)


# ------------------------------------------------
# 6. HOME PAGE
# ------------------------------------------------

if page == "Home":

    st.subheader("Welcome to ShopSense")

    st.write(
        "Describe a shopping session below. "
        "Our Machine Learning model will estimate "
        "whether the visitor is likely to purchase."
    )

    st.info(
        "This is a shopping-session simulator. "
        "In a real e-commerce website, these details "
        "would normally come from website analytics."
    )

    # Summary cards

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "ML Algorithm",
        "Random Forest"
    )

    col2.metric(
        "Input Questions",
        "6"
    )

    col3.metric(
        "Prediction",
        "Purchase / No Purchase"
    )

    st.divider()

    st.subheader("Enter Shopping Session Details")

    st.caption(
        "Answer the following simple questions."
    )

    # ------------------------------------------------
    # 7. USER INPUT FORM
    # ------------------------------------------------

    with st.form("shopping_form"):

        col1, col2 = st.columns(2)

        with col1:

            product_pages = st.number_input(
                "How many product pages were viewed?",
                min_value=0,
                max_value=500,
                value=8
            )

            product_minutes = st.number_input(
                "How many minutes were spent browsing products?",
                min_value=0.0,
                max_value=1000.0,
                value=10.0,
                step=1.0
            )

            information_pages = st.number_input(
                "How many informational pages were viewed?",
                min_value=0,
                max_value=200,
                value=2
            )

        with col2:

            administrative_pages = st.number_input(
                "How many administrative or general pages were viewed?",
                min_value=0,
                max_value=200,
                value=1
            )

            visitor_type = st.selectbox(
                "Is the visitor new or returning?",
                [
                    "New visitor",
                    "Returning visitor",
                    "Other / unknown"
                ]
            )

            session_date = st.date_input(
                "Select the shopping session date",
                value=date.today()
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Purchase Intention",
            use_container_width=True
        )


    # ------------------------------------------------
    # 8. PROCESS USER INPUT AND PREDICT
    # ------------------------------------------------

    if submitted:

        # Convert visitor type into dataset format

        visitor_mapping = {
            "New visitor": "New_Visitor",
            "Returning visitor": "Returning_Visitor",
            "Other / unknown": "Other"
        }

        # Convert date into dataset month format

        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "June",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec"
        }

        month = month_names[session_date.month]

        # Determine whether it is a weekend

        weekend = session_date.weekday() >= 5

        # Create input dataframe

        new_session = pd.DataFrame([{

            "Administrative": administrative_pages,

            "Informational": information_pages,

            "ProductRelated": product_pages,

            "ProductRelated_Duration": product_minutes * 60,

            "Month": month,

            "VisitorType": visitor_mapping[visitor_type],

            "Weekend": str(weekend)

        }])

        # Generate prediction

        prediction = model.predict(new_session)[0]

        probability = model.predict_proba(new_session)[0][1]

        # Show results

        st.divider()

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.success(
                    "✅ Purchase Likely\n\n"
                    "The model predicts that the visitor "
                    "may make a purchase."
                )

            else:

                st.warning(
                    "Purchase Less Likely\n\n"
                    "The model predicts that the visitor "
                    "may not make a purchase."
                )

        with col2:

            st.metric(
                "Estimated Purchase Score",
                f"{probability:.1%}"
            )

            st.progress(float(probability))

        st.caption(
            "This score is a model estimate, not a guarantee "
            "of actual customer purchasing behaviour."
        )

        with st.expander("View information sent to the model"):

            st.dataframe(
                new_session,
                use_container_width=True,
                hide_index=True
            )


# ------------------------------------------------
# 9. ABOUT PAGE
# ------------------------------------------------

else:

    st.subheader("About ShopSense")

    st.write(
        "ShopSense is a beginner-friendly Machine Learning "
        "application that predicts online customer purchase "
        "intention using shopping-session information."
    )

    st.divider()

    st.markdown("### Project Objective")

    st.write(
        "The objective is to understand customer browsing "
        "behaviour and predict whether a shopping session "
        "may result in a purchase."
    )

    st.markdown("### How It Works")

    st.markdown("""
    1. Download historical shopping data.
    2. Prepare the dataset.
    3. Train a Random Forest model.
    4. Evaluate the model.
    5. Accept new shopping-session details.
    6. Generate a purchase prediction.
    """)

    st.markdown("### Technologies Used")

    st.markdown("""
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest
    - Joblib
    - Streamlit
    """)

    st.markdown("### Dataset")

    st.write(
        "UCI Online Shoppers Purchasing Intention Dataset."
    )

    st.markdown(
        "[View Dataset](https://archive.ics.uci.edu/dataset/468/online)"
    )

    st.info(
        "ShopSense learns from a fixed historical dataset "
        "and predicts new shopping sessions using the same "
        "features."
    )
