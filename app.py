import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriYield AI",
    page_icon="🌾",
    layout="centered"
)


# ============================================================
# AGRICULTURE THEME
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #E8F5E9 0%,
            #F7F3E8 50%,
            #EAF4E2 100%
        );
    }

    /* Main content area */
    .main {
        background: transparent;
    }

    /* Title */
    h1 {
        color: #1B5E20;
        font-weight: 800;
    }

    /* Section headings */
    h2, h3 {
        color: #2E7D32;
        font-weight: 700;
    }

    /* Normal text */
    p {
        color: #355E3B;
    }

    /* Input boxes */
    .stSelectbox > div > div,
    .stNumberInput > div > div {
        background-color: #FFFFFF;
        border-radius: 10px;
    }

    /* Prediction button */
    .stButton > button {
        background: linear-gradient(
            135deg,
            #8D6E63,
            #6D4C41
        );
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1rem;
        box-shadow: 0 4px 10px rgba(109, 76, 65, 0.25);
        transition: all 0.2s ease;
    }

    /* Button hover */
    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #6D4C41,
            #5D4037
        );
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(109, 76, 65, 0.3);
    }

    /* Prediction box */
    .stAlert {
        border-radius: 12px;
    }

    /* Divider */
    hr {
        border-color: #A5D6A7;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("agri_model.pkl")


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🌾 AgriYield AI")

st.write(
    "Predict agricultural crop yield using Machine Learning."
)

st.divider()


# ============================================================
# GET FEATURES FROM SAVED MODEL
# ============================================================

preprocessor = model.named_steps["prepr"]

categorical_cols = []
numeric_cols = []

for name, transformer, columns in preprocessor.transformers_:

    if name == "cat":
        categorical_cols = list(columns)

    elif name == "num":
        numeric_cols = list(columns)


# ============================================================
# INPUT DATA
# ============================================================

input_data = {}


# ============================================================
# CROP INFORMATION
# ============================================================

st.subheader("🌱 Crop Information")


# ------------------------------------------------------------
# CATEGORICAL FEATURES
# ------------------------------------------------------------

if len(categorical_cols) > 0:

    cat_pipeline = preprocessor.named_transformers_["cat"]

    encoder = cat_pipeline.named_steps["encoder"]

    for col in categorical_cols:

        column_index = categorical_cols.index(col)

        options = list(
            encoder.categories_[column_index]
        )

        input_data[col] = st.selectbox(
            col,
            options
        )


# ============================================================
# AGRICULTURAL INFORMATION
# ============================================================

st.subheader("📊 Agricultural Information")


for col in numeric_cols:

    # --------------------------------------------------------
    # YEAR DROPDOWN
    # --------------------------------------------------------

    if col == "Year":

        year_options = list(
            range(1961, 2024)
        )

        input_data[col] = st.selectbox(
            "Year",
            year_options
        )


    # --------------------------------------------------------
    # AREA HARVESTED DROPDOWN
    # --------------------------------------------------------

    elif col == "Area Harvested":

        area_harvested_options = [
            100,
            500,
            1000,
            5000,
            10000,
            50000,
            100000,
            500000,
            1000000
        ]

        input_data[col] = st.selectbox(
            "Area Harvested",
            area_harvested_options
        )


    # --------------------------------------------------------
    # OTHER NUMERICAL FEATURES
    # --------------------------------------------------------

    else:

        input_data[col] = st.number_input(
            col,
            value=0.0
        )


# ============================================================
# PREDICT
# ============================================================

st.divider()

if st.button(
    "🌾 Predict Yield",
    use_container_width=True
):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    st.success(
        f"🌾 Predicted Yield: **{prediction:,.2f}**"
    )

    st.info(
        "This is the estimated crop yield based on "
        "the selected agricultural information."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Developed By Habibulie 🟢alias Leda 🔴 | Data Science Student."
)