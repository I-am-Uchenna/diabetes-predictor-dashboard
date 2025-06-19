import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib as mpl

# Enable dark mode styling
st.set_page_config(page_title="Diabetes Risk Dashboard", layout="wide")

# Apply dark theme using mpl
mpl.rcParams.update({
    'axes.facecolor': '#111111',
    'axes.edgecolor': 'white',
    'figure.facecolor': '#111111',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'axes.titlecolor': 'white'
})

@st.cache_data
def load_data():
    df1 = pd.read_csv("https://raw.githubusercontent.com/rajeevratan84/data-analyst-bootcamp/master/labs.csv")
    df2 = pd.read_csv("https://raw.githubusercontent.com/rajeevratan84/data-analyst-bootcamp/master/examination.csv").drop(['SEQN'], axis=1)
    df3 = pd.read_csv("https://raw.githubusercontent.com/rajeevratan84/data-analyst-bootcamp/master/demographic.csv").drop(['SEQN'], axis=1)
    df4 = pd.read_csv("https://raw.githubusercontent.com/rajeevratan84/data-analyst-bootcamp/master/diet.csv").drop(['SEQN'], axis=1)
    df5 = pd.read_csv("https://raw.githubusercontent.com/rajeevratan84/data-analyst-bootcamp/master/questionnaire.csv").drop(['SEQN'], axis=1)

    df = pd.concat([df1, df2, df3, df4, df5], axis=1, join='inner')

    df = df.rename(columns={
        'SEQN': 'ID',
        'RIAGENDR': 'Gender',
        'DMDYRSUS': 'Years_in_US',
        'INDFMPIR': 'Family_income',
        'LBXGH': 'GlycoHemoglobin',
        'BMXARMC': 'ArmCircum',
        'BMDAVSAD': 'SaggitalAbdominal',
        'MGDCGSZ': 'GripStrength',
        'DRABF': 'Breast_fed'
    })

    df = df[['Gender', 'Years_in_US', 'Family_income','GlycoHemoglobin', 'ArmCircum',
             'SaggitalAbdominal', 'GripStrength', 'Breast_fed']]

    df['Years_in_US'] = df['Years_in_US'].apply(lambda x: x if x > 0 else 0)
    df['GlycoHemoglobin'] = df['GlycoHemoglobin'].fillna(df['GlycoHemoglobin'].median())
    df['SaggitalAbdominal'] = df['SaggitalAbdominal'].fillna(df['SaggitalAbdominal'].median())
    df['ArmCircum'] = df['ArmCircum'].fillna(df['ArmCircum'].median())
    df['GripStrength'] = df['GripStrength'].fillna(df['GripStrength'].median())
    df['Family_income'] = df['Family_income'].fillna(method='ffill')
    df['Breast_fed'] = df['Breast_fed'].fillna(value=1)

    df.loc[df['GlycoHemoglobin'] < 6.0, 'Diabetes'] = 0
    df.loc[(df['GlycoHemoglobin'] >= 6.0) & (df['GlycoHemoglobin'] <= 6.4), 'Diabetes'] = 1
    df.loc[df['GlycoHemoglobin'] >= 6.5, 'Diabetes'] = 2

    df = df.drop(['GlycoHemoglobin'], axis=1)
    return df

df = load_data()

st.title("🧠 Diabetes Risk Prediction Dashboard")
st.markdown("""
This dashboard predicts the risk of diabetes using demographic and physical health data from the NHANES dataset.
""")

st.subheader("📈 Feature Correlation")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df.drop('Diabetes', axis=1).corr(), annot=True, cmap='viridis', ax=ax)
st.pyplot(fig, use_container_width=True)

st.subheader("📊 Diabetes Risk Distribution")
col1, col2 = st.columns([2, 1])
with col1:
    st.bar_chart(df['Diabetes'].value_counts().sort_index())

st.subheader("⚙️ Train & Evaluate Models")

X = df.drop('Diabetes', axis=1)
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100),
    "Bagging (DT)": BaggingClassifier(estimator=DecisionTreeClassifier()),
    "Bagging (KNN)": BaggingClassifier(estimator=KNeighborsClassifier()),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = model.score(X_test, y_test)

result_df = pd.DataFrame({"Model": list(results.keys()), "Accuracy": list(results.values())})
st.dataframe(result_df)

st.bar_chart(result_df.set_index("Model"))

st.subheader("🔍 Predict Your Diabetes Risk")

input_data = {
    'Gender': st.selectbox("Gender (1 = Male, 2 = Female)", [1, 2], help="Choose the gender: 1 for Male, 2 for Female"),
    'Years_in_US': st.slider("Years lived in the US", 0, 80, 10, help="Number of years the person has lived in the United States"),
    'Family_income': st.slider("Family Income Ratio", 0.0, 5.0, 2.5, help="Income-to-poverty ratio (higher = higher income)"),
    'ArmCircum': st.slider("Arm Circumference (cm)", 15.0, 50.0, 30.0, help="Measurement around the upper arm in centimeters"),
    'SaggitalAbdominal': st.slider("Sagittal Abdominal Diameter (cm)", 10.0, 50.0, 25.0, help="Abdominal height while lying down"),
    'GripStrength': st.slider("Grip Strength (kg)", 10.0, 100.0, 50.0, help="Hand grip strength in kilograms"),
    'Breast_fed': st.selectbox("Were you breastfed as a baby? (1 = Yes, 0 = No)", [1, 0], help="1 means Yes, 0 means No")
}

user_input = pd.DataFrame([input_data])
selected_model = st.selectbox("Select model for prediction", list(models.keys()))

if st.button("Predict"):
    prediction = models[selected_model].predict(user_input)
    labels = {0: "Normal", 1: "At Risk", 2: "Diabetic"}
    st.success(f"Predicted Risk Category: {labels[int(prediction[0])]}")
