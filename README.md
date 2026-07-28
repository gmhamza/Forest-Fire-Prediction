# 🔥 Forest Fire Risk Prediction and Early Warning System

A Machine Learning based system that predicts the risk of a forest fire from
daily weather and Fire Weather Index (FWI) readings, and issues an early
warning through an interactive Streamlit web app.

Built as part of a **Data Science Internship mini project** (B.E. 5th Semester).

---

## 📌 Project Description

Forest fires cause massive loss of wildlife, vegetation, property, and human
life every year, and are often detected only after they have already spread.
This project uses historical weather data (temperature, humidity, wind speed,
rainfall) combined with the **Fire Weather Index (FWI) system** (FFMC, DMC,
DC, ISI, BUI, FWI) to train a Machine Learning classification model that
predicts whether a given day is at risk of a forest fire (`fire`) or not
(`not fire`), so that preventive action can be taken early.

## ✨ Features

- Cleaned and analyzed real-world-style forest fire weather dataset
- Full Exploratory Data Analysis (EDA) notebook with 15+ visualizations
- 6 Machine Learning algorithms trained and compared
- Best model automatically selected and saved using Pickle
- Interactive **Streamlit web app** for live risk prediction
- Color-coded risk result with confidence score and safety recommendations
- Model comparison dashboard inside the app

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Persistence | Pickle |
| Web App | Streamlit |

## 📂 Folder Structure

```
ForestFirePrediction/
│
├── app.py                     # Streamlit web application
├── train_model.py             # Data preprocessing + model training script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   └── Algerian_forest_fires_dataset.csv        # Dataset used for training
│
├── models/
│   ├── forest_fire_model.pkl      # Best trained model (Logistic Regression)
│   ├── scaler.pkl                 # StandardScaler used on features
│   ├── region_encoder.pkl         # LabelEncoder for Region column
│   ├── target_encoder.pkl         # LabelEncoder for target (fire/not fire)
│   ├── feature_columns.pkl        # Feature column order
│   └── model_comparison_results.csv
│
├── notebooks/
│   ├── EDA_Forest_Fire.ipynb  # Full EDA notebook (Colab / Jupyter ready)
│   └── Algerian_forest_fires_dataset.csv
│
```

## ⚙️ Installation

1. Clone / download this project folder.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ How to Run

**Step 1 — (Optional) Regenerate the dataset**
```bash
python generate_dataset.py
```

**Step 2 — Train the models**
```bash
python train_model.py
```
This prints accuracy/precision/recall/F1 for all 6 models and saves the
best one to the `models/` folder.

**Step 3 — Run the Streamlit app**
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`.

**To run the EDA notebook:** open `notebooks/EDA_Forest_Fire.ipynb` in
Jupyter Notebook, JupyterLab, or upload it to Google Colab (along with
`Algerian_forest_fires_dataset.csv` from the same folder).

## 📊 Output

- Console output of model training with accuracy, precision, recall, F1-score,
  confusion matrix and classification report for all 6 algorithms
- `models/model_comparison_results.csv` — comparison table
- Interactive Streamlit app with a Predict Risk page, Model Info dashboard,
  and About page

## 🚀 Future Improvements

- Integrate a live weather API (e.g. OpenWeatherMap) for real-time prediction
- Add satellite/IoT sensor data for more accurate ground-level detection
- Send automatic SMS/Email alerts to the forest department when high risk is detected
- Add a live map showing fire-risk zones
- Deploy the app on the cloud (Streamlit Community Cloud / Render / Heroku)

## 👤 Gulam Mahmood Hamza

Data Science Internship Project — B.E. 5th Semester
