import streamlit as st
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MLProject",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# CUSTOM PAGE STYLING
# ==============================

st.markdown("""
<style>

    /* Main page background */
    .stApp {
        background-color: #f5f9ff;
    }

    /* Header area */
    .main-title {
        letter-spacing: -0.5px;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1400px;
    }

    /* Main title */
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #0f3d91;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .sub-title {
        font-size: 17px;
        color: #5f6b7a;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 650;
        color: #123f8c;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #dbe7f5;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0px 4px 12px rgba(30, 80, 150, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #5f6b7a;
        font-weight: 500;
    }

    [data-testid="stMetricValue"] {
        color: #0f3d91;
        font-weight: 700;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border: 1px solid #cbd9eb;
        background-color: white;
    }

    /* Date input */
    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        background-color: #1769e0;
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background-color: #0f4fb8;
        color: white;
        border: none;
    }

    /* Information boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Horizontal divider */
    hr {
        border: none;
        height: 1px;
        background-color: #dbe7f5;
        margin: 25px 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dbe7f5;
    }

</style>
""", unsafe_allow_html=True)
# =========================================================
# FILE PATHS
# =========================================================

WALMART_MODEL_PATH = "models/walmart_lightgbm_model.pkl"

FMCG_MODEL_PATH = "models/fmcg_xgboost_model.json"

FMCG_PREPROCESSOR_PATH = "models/fmcg_preprocessor.pkl"

FMCG_DATA_PATH = "data/fmcg_sales_3years_1M_rows.csv"

WALMART_DATA_PATH = "data/Walmart.csv"


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    # Walmart LightGBM
    walmart_model = joblib.load(
        WALMART_MODEL_PATH
    )

    # FMCG XGBoost
    fmcg_model = XGBRegressor()

    fmcg_model.load_model(
        FMCG_MODEL_PATH
    )

    # FMCG preprocessor
    fmcg_preprocessor = joblib.load(
        FMCG_PREPROCESSOR_PATH
    )

    return (
        walmart_model,
        fmcg_model,
        fmcg_preprocessor
    )


# =========================================================
# LOAD FMCG DATA
# =========================================================

@st.cache_data
def load_fmcg_data():

    df = pd.read_csv(
        FMCG_DATA_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


# =========================================================
# LOAD WALMART DATA
# =========================================================

@st.cache_data
def load_walmart_data():

    df = pd.read_csv(
        WALMART_DATA_PATH
    )

    return df


# =========================================================
# CREATE FMCG PREDICTION FEATURES
# =========================================================

def create_fmcg_prediction_features(
    df,
    store_id,
    sku_id,
    prediction_date
):

    data = df.copy()

    # -----------------------------------------------------
    # Convert date
    # -----------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"]
    )

    # -----------------------------------------------------
    # Filter Store + SKU
    # -----------------------------------------------------

    product_history = data[
        (data["store_id"] == store_id) &
        (data["sku_id"] == sku_id)
    ].copy()

    if product_history.empty:
        return None

    # -----------------------------------------------------
    # Sort history
    # -----------------------------------------------------

    product_history = product_history.sort_values(
        "date"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Require sufficient history
    # -----------------------------------------------------

    if len(product_history) < 28:

        return None

    # -----------------------------------------------------
    # Latest historical record
    # -----------------------------------------------------

    latest_record = product_history.iloc[-1].copy()

    prediction_date = pd.to_datetime(
        prediction_date
    )

    # -----------------------------------------------------
    # Create prediction row
    # -----------------------------------------------------

    prediction_row = latest_record.copy()

    prediction_row["date"] = prediction_date

    # -----------------------------------------------------
    # Date features
    # -----------------------------------------------------

    prediction_row["year"] = prediction_date.year

    prediction_row["month"] = prediction_date.month

    prediction_row["day"] = prediction_date.day

    prediction_row["weekofyear"] = (
        prediction_date.isocalendar().week
    )

    prediction_row["weekday"] = (
        prediction_date.weekday()
    )

    prediction_row["is_weekend"] = int(
        prediction_date.weekday() >= 5
    )

    # -----------------------------------------------------
    # Holiday
    # -----------------------------------------------------

    prediction_row["is_holiday"] = 0

    # -----------------------------------------------------
    # Historical sales
    # -----------------------------------------------------

    sales_history = (
        product_history["units_sold"]
        .dropna()
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Lag features
    # -----------------------------------------------------

    prediction_row["Lag_1"] = (
        sales_history.iloc[-1]
    )

    prediction_row["Lag_7"] = (
        sales_history.iloc[-7]
    )

    prediction_row["Lag_14"] = (
        sales_history.iloc[-14]
    )

    prediction_row["Lag_28"] = (
        sales_history.iloc[-28]
    )

    # -----------------------------------------------------
    # Rolling features
    # -----------------------------------------------------

    prediction_row["Rolling_Mean_7"] = (
        sales_history.tail(7).mean()
    )

    prediction_row["Rolling_Mean_14"] = (
        sales_history.tail(14).mean()
    )

    prediction_row["Rolling_Mean_28"] = (
        sales_history.tail(28).mean()
    )

    prediction_row["Rolling_Std_7"] = (
        sales_history.tail(7).std()
    )

    # -----------------------------------------------------
    # Price features
    # -----------------------------------------------------

    prediction_row["discount_amount"] = (
        prediction_row["list_price"]
        * prediction_row["discount_pct"]
        / 100
    )

    prediction_row["selling_price"] = (
        prediction_row["list_price"]
        - prediction_row["discount_amount"]
    )

    prediction_row["promo_discount"] = (
        prediction_row["discount_pct"]
        * prediction_row["promo_flag"]
    )

    # -----------------------------------------------------
    # Inventory features
    # -----------------------------------------------------

    prediction_row["low_stock_flag"] = int(
        prediction_row["stock_on_hand"] <= 20
    )

    prediction_row["inventory_cover_7"] = (
        prediction_row["stock_on_hand"]
        /
        (prediction_row["Rolling_Mean_7"] + 1e-6)
    )

    prediction_row["inventory_cover_28"] = (
        prediction_row["stock_on_hand"]
        /
        (prediction_row["Rolling_Mean_28"] + 1e-6)
    )

    prediction_row["low_inventory_cover_flag"] = int(
        prediction_row["inventory_cover_7"] < 3
    )

    return prediction_row


# =========================================================
# FMCG MODEL PREDICTION
# =========================================================

def predict_fmcg_demand(
    df,
    model,
    preprocessor,
    store_id,
    sku_id,
    prediction_date
):

    prediction_row = create_fmcg_prediction_features(
        df,
        store_id,
        sku_id,
        prediction_date
    )

    if prediction_row is None:

        return None

    # -----------------------------------------------------
    # Features used during training
    # -----------------------------------------------------

    features = [

        "year",
        "month",
        "day",
        "weekofyear",
        "weekday",
        "is_weekend",
        "is_holiday",

        "store_id",
        "sku_id",
        "category",
        "subcategory",
        "brand",
        "channel",

        "temperature",
        "rain_mm",

        "list_price",
        "discount_pct",
        "selling_price",

        "promo_flag",
        "promo_discount",

        "stock_on_hand",
        "stock_out_flag",
        "lead_time_days",

        "low_stock_flag",

        "inventory_cover_7",
        "inventory_cover_28",
        "low_inventory_cover_flag",

        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Lag_28",

        "Rolling_Mean_7",
        "Rolling_Mean_14",
        "Rolling_Mean_28",
        "Rolling_Std_7"
    ]

    # -----------------------------------------------------
    # Create model input
    # -----------------------------------------------------

    X = pd.DataFrame(
        [prediction_row[features]]
    )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    X_encoded = preprocessor.transform(
        X
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(
        X_encoded
    )[0]

    prediction = max(
        0,
        float(prediction)
    )

    return prediction

# =========================================================
# INVENTORY CALCULATIONS
# =========================================================

def calculate_inventory_metrics(
    df,
    store_id,
    sku_id,
    predicted_units
):

    # -----------------------------------------------------
    # Get product history
    # -----------------------------------------------------

    product_history = df[
        (df["store_id"] == store_id) &
        (df["sku_id"] == sku_id)
    ].copy()

    if product_history.empty:
        return None

    # Sort by date
    product_history = product_history.sort_values(
        "date"
    )

    # -----------------------------------------------------
    # Latest inventory information
    # -----------------------------------------------------

    latest = product_history.iloc[-1]

    stock_on_hand = float(
        latest["stock_on_hand"]
    )

    lead_time_days = float(
        latest["lead_time_days"]
    )

    # -----------------------------------------------------
    # Estimate forecast error
    # -----------------------------------------------------

    # Use historical demand variability
    sales_history = (
        product_history["units_sold"]
        .dropna()
    )

    if len(sales_history) >= 7:

        error_std = float(
            sales_history.tail(28).std()
        )

    else:

        error_std = float(
            sales_history.std()
        )

    # Safety fallback
    if pd.isna(error_std):

        error_std = 0.0

    # -----------------------------------------------------
    # 95% service level
    # -----------------------------------------------------

    Z_VALUE = 1.645

    # -----------------------------------------------------
    # Safety Stock
    # -----------------------------------------------------

    safety_stock = (
        Z_VALUE
        * error_std
        * np.sqrt(lead_time_days)
    )

    # -----------------------------------------------------
    # Demand during lead time
    # -----------------------------------------------------

    demand_during_lead_time = (
        predicted_units
        * lead_time_days
    )

    # -----------------------------------------------------
    # Reorder Point
    # -----------------------------------------------------

    reorder_point = (
        demand_during_lead_time
        + safety_stock
    )

    # -----------------------------------------------------
    # Recommended Order
    # -----------------------------------------------------

    recommended_order = np.ceil(
        max(
            reorder_point - stock_on_hand,
            0
        )
    )

    # -----------------------------------------------------
    # Inventory Coverage
    # -----------------------------------------------------

    inventory_coverage_days = (
        stock_on_hand
        /
        (predicted_units + 1e-6)
    )

    # -----------------------------------------------------
    # Inventory Status
    # -----------------------------------------------------

    if stock_on_hand <= 0:

        inventory_status = "Out of Stock"

    elif inventory_coverage_days < lead_time_days:

        inventory_status = "Critical"

    elif stock_on_hand < reorder_point:

        inventory_status = "Restock Required"

    elif inventory_coverage_days < lead_time_days + 2:

        inventory_status = "Low Stock"

    else:

        inventory_status = "Sufficient Stock"

    # -----------------------------------------------------
    # Inventory Risk Score
    # -----------------------------------------------------

    inventory_risk_score = (
        100
        * (
            1
            -
            stock_on_hand
            /
            (reorder_point + 1e-6)
        )
    )

    inventory_risk_score = np.clip(
        inventory_risk_score,
        0,
        100
    )

    # -----------------------------------------------------
    # Risk Level
    # -----------------------------------------------------

    if inventory_risk_score >= 80:

        risk_level = "Critical Risk"

    elif inventory_risk_score >= 60:

        risk_level = "High Risk"

    elif inventory_risk_score >= 40:

        risk_level = "Medium Risk"

    elif inventory_risk_score >= 20:

        risk_level = "Low Risk"

    else:

        risk_level = "Very Low Risk"

    # -----------------------------------------------------
    # Recommendation
    # -----------------------------------------------------

    if inventory_status == "Out of Stock":

        recommendation = "Immediate Restock"

    elif inventory_status == "Critical":

        recommendation = "Urgent Restock"

    elif inventory_status == "Restock Required":

        recommendation = "Place Order"

    elif inventory_status == "Low Stock":

        recommendation = "Monitor Stock"

    else:

        recommendation = "No Action Required"

    # -----------------------------------------------------
    # Return all metrics
    # -----------------------------------------------------

    return {
        "stock_on_hand": stock_on_hand,
        "lead_time_days": lead_time_days,
        "error_std": error_std,
        "safety_stock": safety_stock,
        "demand_during_lead_time": demand_during_lead_time,
        "reorder_point": reorder_point,
        "recommended_order": recommended_order,
        "inventory_coverage_days": inventory_coverage_days,
        "inventory_status": inventory_status,
        "inventory_risk_score": inventory_risk_score,
        "risk_level": risk_level,
        "recommendation": recommendation
    }
# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    walmart_model, fmcg_model, fmcg_preprocessor = load_models()

    fmcg_df = load_fmcg_data()

    walmart_df = load_walmart_data()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "Unable to load the application resources."
    )

    st.exception(e)


# =========================================================
# MAIN APPLICATION
# =========================================================

if models_loaded:

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        '<div class="main-title">📊 Sales Forecasting & Inventory Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">AI-powered demand forecasting and inventory decision support system</div>',
        unsafe_allow_html=True
    )

    st.divider()


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "FMCG Model Accuracy",
            "76.85%"
        )

    with col2:
        st.metric(
            "Walmart Model Accuracy",
            "98.89%"
        )

    st.write("")

    # =====================================================
    # PREDICTION SECTION
    # =====================================================

    st.markdown('<div class="section-title">🔮 Demand Prediction</div>', unsafe_allow_html=True)

    st.write(
        "Select a store, product, and prediction date."
    )


    # =====================================================
    # INPUTS
    # =====================================================

    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # STORE
    # -----------------------------------------------------

    with col1:

        store_options = sorted(
            fmcg_df["store_id"]
            .dropna()
            .unique()
        )

        store = st.selectbox(
            "Store",
            store_options
        )


    # -----------------------------------------------------
    # PRODUCT
    # -----------------------------------------------------
    with col2:

        product_options = sorted(
        fmcg_df[
            fmcg_df["store_id"] == store
        ]["sku_id"]
        .dropna()
        .unique()
    )

    product = st.selectbox(
        "Product / SKU",
        product_options
    )


    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    with col3:

        prediction_date = st.date_input(
            "Prediction Date"
        )


    st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🚀 Predict Demand",
    use_container_width=True
):

    try:

        # -------------------------------------------------
        # Demand prediction
        # -------------------------------------------------

        prediction = predict_fmcg_demand(
            fmcg_df,
            fmcg_model,
            fmcg_preprocessor,
            store,
            product,
            prediction_date
        )

        if prediction is None:

            st.error(
                "Insufficient historical data for "
                "the selected store and product."
            )

        else:

            # -------------------------------------------------
            # Inventory calculation
            # -------------------------------------------------

            inventory_metrics = calculate_inventory_metrics(
                fmcg_df,
                store,
                product,
                prediction
            )

            if inventory_metrics is None:

                st.error(
                    "Inventory information could not be found."
                )

            else:

                # ---------------------------------------------
                # Save prediction
                # ---------------------------------------------

                st.session_state["prediction"] = prediction

                st.session_state["inventory_metrics"] = (
                    inventory_metrics
                )

                st.session_state["selected_store"] = store

                st.session_state["selected_product"] = product

                st.session_state["prediction_date"] = (
                    prediction_date
                )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)
# =========================================================
# DISPLAY RESULTS
# =========================================================

if "prediction" in st.session_state:

    st.divider()

    st.markdown('<div class="section-title">📈 Forecast & Inventory Analysis</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # Retrieve results
    # -----------------------------------------------------

    prediction = st.session_state[
        "prediction"
    ]

    inventory = st.session_state[
        "inventory_metrics"
    ]

    selected_store = st.session_state[
        "selected_store"
    ]

    selected_product = st.session_state[
        "selected_product"
    ]

    selected_date = st.session_state[
        "prediction_date"
    ]


    # =====================================================
    # BASIC INFORMATION
    # =====================================================

    st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Store",
            selected_store
        )

    with col2:

        st.metric(
            "Product",
            selected_product
        )

    with col3:

        st.metric(
            "Forecast Date",
            str(selected_date)
        )

    with col4:

        st.metric(
            "Predicted Demand",
            f"{prediction:.0f} units"
        )


    st.divider()


    # =====================================================
    # INVENTORY METRICS
    # =====================================================

    st.markdown('<div class="section-title">📦 Inventory Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Stock",
            f"{inventory['stock_on_hand']:.0f}"
        )

    with col2:

        st.metric(
            "Lead Time",
            f"{inventory['lead_time_days']:.0f} days"
        )

    with col3:

        st.metric(
            "Safety Stock",
            f"{inventory['safety_stock']:.0f}"
        )

    with col4:

        st.metric(
            "Reorder Point",
            f"{inventory['reorder_point']:.0f}"
        )


    st.write("")


    # =====================================================
    # ORDER RECOMMENDATION
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Demand During Lead Time",
            f"{inventory['demand_during_lead_time']:.0f}"
        )

    with col2:

        st.metric(
            "Recommended Order",
            f"{inventory['recommended_order']:.0f}"
        )

    with col3:

        st.metric(
            "Stock Coverage",
            f"{inventory['inventory_coverage_days']:.1f} days"
        )


    st.divider()


    # =====================================================
    # STATUS AND RISK
    # =====================================================

    st.markdown('<div class="section-title">⚠️ Inventory Status</div>', unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)


    with col1:

        status = inventory[
            "inventory_status"
        ]

        if status == "Out of Stock":

            st.error(
                f"🔴 {status}"
            )

        elif status == "Critical":

            st.error(
                f"🔴 {status}"
            )

        elif status == "Restock Required":

            st.warning(
                f"🟠 {status}"
            )

        elif status == "Low Stock":

            st.warning(
                f"🟡 {status}"
            )

        else:

            st.success(
                f"🟢 {status}"
            )


    with col2:

        risk = inventory[
            "risk_level"
        ]

        risk_score = inventory[
            "inventory_risk_score"
        ]

        st.metric(
            "Risk Level",
            risk
        )

        st.progress(
            int(risk_score) / 100
        )

        st.caption(
            f"Risk Score: {risk_score:.1f}/100"
        )


    with col3:

        recommendation = inventory[
            "recommendation"
        ]

        if recommendation in [
            "Immediate Restock",
            "Urgent Restock"
        ]:

            st.error(
                f"🚨 {recommendation}"
            )

        elif recommendation == "Place Order":

            st.warning(
                f"📦 {recommendation}"
            )

        elif recommendation == "Monitor Stock":

            st.info(
                f"👀 {recommendation}"
            )

        else:

            st.success(
                f"✅ {recommendation}"
            )


    st.divider()


    # =====================================================
    # DECISION SUMMARY
    # =====================================================

    st.markdown('<div class="section-title">💡 Inventory Decision</div>', unsafe_allow_html=True)

    if inventory["recommended_order"] > 0:

        st.write(
            f"Based on the predicted demand of "
            f"**{prediction:.0f} units** and the current "
            f"inventory level of "
            f"**{inventory['stock_on_hand']:.0f} units**, "
            f"the system recommends ordering "
            f"**{inventory['recommended_order']:.0f} units**."
        )

    else:

        st.write(
            "Current inventory is sufficient based on "
            "the predicted demand and lead time."
        )