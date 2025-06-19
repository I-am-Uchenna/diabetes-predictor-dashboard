# 🧠 Diabetes Risk Prediction Dashboard

This interactive Streamlit dashboard uses machine learning models to predict a person's diabetes risk based on demographic and body measurement data from the NHANES 2013–2014 dataset.

---

## 📌 Executive Summary
This analysis used data from the 2013–2014 NHANES survey to predict an individual’s risk of diabetes using demographic information and physical body metrics. Several machine learning models were evaluated to determine which approach provides the most accurate classification into three categories: Normal, At-Risk, and Diabetic.

---

## 🔍 Key Clinical Insights

### Physical Measurements Matter
Individuals with higher **abdominal fat** (sagittal abdominal diameter), larger **arm circumference**, and reduced **grip strength** showed a higher likelihood of being diabetic.

### Glycohemoglobin (HbA1c) Benchmark
- **<6.0%** → Normal
- **6.0–6.4%** → At-Risk
- **≥6.5%** → Diabetic

### Model Performance
The most effective models were:
- ✅ AdaBoost
- ✅ Bagging (Decision Trees)
- ✅ Multi-layer Perceptron (MLP)

All three achieved approximately **91% prediction accuracy**.

### Socioeconomic and Historical Factors
Income level and whether an individual was breastfed were included, but showed **weaker correlation** to diabetes status.

---

## 🧠 Recommendation
For practical deployments (e.g., in digital health apps or screening tools), **ensemble models** such as **AdaBoost** or **MLPs** should be used. These models reliably classify individuals into risk groups using readily available **non-invasive measurements**.

---

## 📂 Dataset

### Diabetes Prediction using demographic data and body measurements

The [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/Nchs/Nhanes/about_nhanes.htm) is a program of studies designed to assess the health and nutritional status of adults and children in the United States. The survey combines interviews and physical examinations and is run by the CDC’s National Center for Health Statistics (NCHS).

The NHANES program began in the early 1960s and became a continuous program in 1999. Each year, a nationally representative sample of about 5,000 persons is examined across various U.S. counties.

The NHANES 2013–2014 datasets include:

### 1. Demographics dataset
- Demographic and socioeconomic variables
- [Variable dictionary](https://wwwn.cdc.gov/Nchs/Nhanes/Search/variablelist.aspx?Component=Demographics&CycleBeginYear=2013)

### 2. Examinations dataset
- Blood pressure, body measures, grip strength, oral health, taste & smell
- [Variable dictionary](https://wwwn.cdc.gov/Nchs/Nhanes/Search/variablelist.aspx?Component=Examination&CycleBeginYear=2013)

### 3. Dietary data
- Total nutrient intake (1st day)
- [Variable dictionary](https://wwwn.cdc.gov/Nchs/Nhanes/Search/variablelist.aspx?Component=Dietary&CycleBeginYear=2013)

### 4. Laboratory dataset
Includes (partial list):
- Glycohemoglobin (HbA1c)
- Cholesterol panels (HDL, LDL, Total)
- Plasma fasting glucose, insulin
- Heavy metals and nutrients (zinc, copper, selenium)
- Hepatitis, HPV, HIV markers
- [Variable dictionary](https://wwwn.cdc.gov/Nchs/Nhanes/Search/variablelist.aspx?Component=Laboratory&CycleBeginYear=2013)

### 5. Questionnaire dataset
Covers multiple health and behavioral areas:
- Medical history, physical activity, mental health, immunization, drug use, smoking
- [Variable dictionary](https://wwwn.cdc.gov/Nchs/Nhanes/Search/variablelist.aspx?Component=Questionnaire&CycleBeginYear=2013)

### 6. Medication dataset
- Includes prescription medication data
- [Variable dictionary](https://wwwn.cdc.gov/Nchs/Nhanes/Search/variablelist.aspx?Component=Questionnaire&CycleBeginYear=2013)
Data is sourced from the [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/), combining:
- Demographics
- Body examination
- Diet & questionnaire responses
- Lab results (Glycohemoglobin)

---

## 🛠️ How to Run the App Locally

1. **Clone this repository**
   ```bash
   git clone https://github.com/I-am-Uchenna/diabetes-predictor-dashboard.git
   cd diabetes-predictor-dashboard
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run diabetes_dashboard.py
   ```

---

## 🧠 Machine Learning Models Used

| Model                  | Description                       |
|------------------------|------------------------------------|
| Linear Regression      | Baseline model                    |
| AdaBoost               | Ensemble using decision trees     |
| Bagging (DecisionTree) | Bootstrapped decision tree models |
| Bagging (KNN)          | Bootstrapped k-nearest neighbors  |
| MLP Neural Network     | Deep learning classification      |

---

## 📊 Diabetes Risk Categories

- `0` → Normal  
- `1` → At Risk  
- `2` → Diabetic  

These categories are based on Glycohemoglobin (HbA1c) levels.

---

## 💡 Credits

Built by Uchenna Ejike using open NHANES data, Streamlit, and Scikit-Learn.

---

## 🌐 Live Demo

> Live via [Streamlit Cloud](https://diabetes-predictor-dashboard.streamlit.app/)
