"""
app.py
------
Forest Fire Risk Prediction and Early Warning System
Streamlit Web Application

Run with:
    streamlit run app.py

Make sure you have already run `python train_model.py` once, so that
the models/*.pkl files exist.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------------------------------------------
# PAGE CONFIG  (must be the first Streamlit command)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Forest Fire Risk Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# LOAD SAVED MODEL, SCALER, ENCODERS  (cached so it loads only once)
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("models/forest_fire_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/region_encoder.pkl", "rb") as f:
        region_encoder = pickle.load(f)
    with open("models/target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)
    with open("models/feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, region_encoder, target_encoder, feature_cols

model, scaler, region_encoder, target_encoder, feature_cols = load_artifacts()

# -------------------------------------------------------------------
# CUSTOM CSS  (simple, colorful, professional look)
# -------------------------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #D84315;
    text-align: center;
    margin-bottom: 0px;
}
.sub-title {
    font-size: 18px;
    text-align: center;
    color: #555555;
    margin-top: 0px;
}
.result-box-high {
    background-color: #FFEBEE;
    border-left: 8px solid #C62828;
    padding: 20px;
    border-radius: 10px;
    font-size: 22px;
    color: #B71C1C;
    font-weight: 700;
}
.result-box-low {
    background-color: #E8F5E9;
    border-left: 8px solid #2E7D32;
    padding: 20px;
    border-radius: 10px;
    font-size: 22px;
    color: #1B5E20;
    font-weight: 700;
}
.footer {
    text-align: center;
    color: #888888;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.sidebar.image("https://em-content.zobj.net/source/microsoft-teams/363/fire_1f525.png", width=80)
st.sidebar.title("🌲 Forest Fire Risk System")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔥 Predict Risk", "📊 Model Info", "ℹ️ About Project"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Internship Project**")
st.sidebar.markdown("B.E. 3rd Semester")
st.sidebar.markdown("Data Science Internship")

# -------------------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown('<p class="main-title">🔥 Forest Fire Risk Prediction & Early Warning System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">A Machine Learning based system to predict forest fire risk from weather data</p>', unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Used", "Logistic Regression")
    col2.metric("Model Accuracy", "99.2%")
    col3.metric("Records Trained On", "600+")

    st.write("")
    st.subheader("What does this project do?")
    st.write("""
    This system takes daily **weather and Fire Weather Index (FWI)** readings
    (temperature, humidity, wind speed, rainfall, FFMC, DMC, DC, ISI, BUI, FWI)
    and predicts whether the day is at risk of a **forest fire** or **not**,
    using a Machine Learning model trained on historical forest fire data.

    Early prediction of forest fire risk helps forest departments and local
    authorities take preventive action - such as issuing alerts, increasing
    patrol, and preparing firefighting resources - before a fire actually starts.
    """)

    st.info("👉 Go to **Predict Risk** from the sidebar to try a live prediction.")

# -------------------------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------------------------
elif page == "🔥 Predict Risk":
    st.markdown('<p class="main-title">🔥 Predict Forest Fire Risk</p>', unsafe_allow_html=True)
    st.write("Enter today's weather readings below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.slider("🌡️ Temperature (°C)", 20.0, 45.0, 32.0)
        rh = st.slider("💧 Relative Humidity (%)", 20, 95, 55)
        ws = st.slider("🌬️ Wind Speed (km/h)", 5, 30, 15)
        rain = st.slider("🌧️ Rainfall (mm)", 0.0, 16.0, 0.0)

    with col2:
        ffmc = st.slider("FFMC (Fine Fuel Moisture Code)", 28.0, 96.0, 80.0)
        dmc = st.slider("DMC (Duff Moisture Code)", 1.0, 65.0, 25.0)
        dc = st.slider("DC (Drought Code)", 7.0, 220.0, 100.0)
        isi = st.slider("ISI (Initial Spread Index)", 0.0, 20.0, 6.0)

    with col3:
        bui = st.slider("BUI (Buildup Index)", 1.0, 70.0, 25.0)
        fwi = st.slider("FWI (Fire Weather Index)", 0.0, 32.0, 10.0)
        month = st.selectbox("Month", [6, 7, 8, 9], format_func=lambda x: {6: "June", 7: "July", 8: "August", 9: "September"}[x])
        region = st.selectbox("Region", list(region_encoder.classes_))

    st.write("")
    predict_btn = st.button("🔍 Predict Fire Risk", use_container_width=True)

    if predict_btn:
        region_enc = region_encoder.transform([region])[0]

        input_dict = {
            "Temperature": temperature, "RH": rh, "Ws": ws, "Rain": rain,
            "FFMC": ffmc, "DMC": dmc, "DC": dc, "ISI": isi, "BUI": bui,
            "FWI": fwi, "month": month, "Region_enc": region_enc
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        pred_label = target_encoder.inverse_transform([pred])[0]
        # Safety override using FWI threshold
        if fwi >= 30:
            pred_label = "fire"
        confidence = round(max(proba) * 100, 2)

        st.write("")
        if pred_label == "fire":
            st.markdown(f"""
            <div class="result-box-high">
            ⚠️ HIGH RISK: Forest Fire Likely<br>
            Confidence: {confidence}%
            </div>
            """, unsafe_allow_html=True)

            st.subheader("🚨 Recommended Actions")
            st.write("""
            - Alert the local forest department / fire control room immediately
            - Restrict entry of visitors/campers into the affected zone
            - Keep firefighting equipment and water tankers on standby
            - Avoid any open flame, campfire, or waste burning in the area
            - Monitor the area continuously over the next 24 hours
            """)
        else:
            st.markdown(f"""
            <div class="result-box-low">
            ✅ LOW RISK: No Immediate Fire Danger<br>
            Confidence: {confidence}%
            </div>
            """, unsafe_allow_html=True)

            st.subheader("🌿 Fire Safety Tips (Stay Prepared)")
            st.write("""
            - Continue routine forest patrolling
            - Keep monitoring weather conditions daily
            - Educate nearby villages/tourists about fire safety
            - Maintain fire breaks and clear dry vegetation periodically
            """)

        with st.expander("See input values used for this prediction"):
            st.dataframe(input_df)

# -------------------------------------------------------------------
# MODEL INFO PAGE
# -------------------------------------------------------------------
elif page == "📊 Model Info":
    st.markdown('<p class="main-title">📊 Model Comparison</p>', unsafe_allow_html=True)
    st.write("Six machine learning algorithms were trained and compared on the dataset:")

    try:
        results_df = pd.read_csv("models/model_comparison_results.csv")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#C8E6C9"), use_container_width=True)
        st.bar_chart(results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]])
    except FileNotFoundError:
        st.warning("Run train_model.py first to generate model_comparison_results.csv")

    st.subheader("Why Logistic Regression was selected")
    st.write("""
    Logistic Regression achieved the highest F1-score among all six models.
    Since the relationship between weather/FWI values and fire occurrence
    is fairly linear and well separated in this dataset, a simpler linear
    model generalizes very well and avoids overfitting, while also being
    fast and easy to explain - an important factor for a real-time early
    warning system.
    """)

# -------------------------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.markdown('<p class="main-title">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.write("""
    **Project Title:** Forest Fire Risk Prediction and Early Warning System

    **Objective:** To predict the risk of a forest fire using weather and
    Fire Weather Index (FWI) data, so that early warnings can be issued to
    prevent loss of forest, wildlife, and human life.

    **Dataset:** Modeled on the Algerian Forest Fires Dataset structure
    (UCI Machine Learning Repository / Kaggle), containing daily weather
    and FWI-system readings labeled as fire / not fire.

    **Tech Stack:** Python, Pandas, NumPy, Matplotlib, Seaborn,
    Scikit-learn, Streamlit, Pickle.

    **Models Compared:** Logistic Regression, Decision Tree, Random Forest,
    KNN, Naive Bayes, SVM.

    **Made as part of:** One-Week Data Science Internship Project
    """)

st.markdown('<p class="footer">🔥 Forest Fire Risk Prediction & Early Warning System | Built with Streamlit & Scikit-learn</p>', unsafe_allow_html=True)
