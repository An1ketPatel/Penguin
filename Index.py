import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Penguin Prediction App")

st.sidebar.header("User Input Features")

uploaded_file = st.sidebar.file_uploader(
    "Upload your input csv file", type=['csv']
)

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island", ('Biscoe','Torgersen','Dream'))
        sex = st.sidebar.selectbox("Sex", ('Male','Female'))
        bill_length_mm = st.sidebar.slider("Bill Length (mm)", 32.1, 59.6)
        bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", 13.1, 21.5)
        flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 172.0, 271.0)
        body_mass_g = st.sidebar.slider("Body Mass (g)", 2708.0, 6300.0)

        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

# Load training structure
penguins_raw = pd.read_csv("penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=['species'])

# Combine input with training data
df = pd.concat([input_df, penguins], axis=0)

# Encode categorical features EXACTLY like training
encode = ["sex", "island"]
for col in encode:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[col], inplace=True)

# Keep only the first row (user input)
df = df.iloc[:1]

st.subheader("User Input Features")
st.write(df)

# Load model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# 🔥 ALIGN FEATURES (MOST IMPORTANT LINE)
df = df.reindex(columns=load_clf.feature_names_in_, fill_value=0)

# Predict
prediction = load_clf.predict(df)
probability = load_clf.predict_proba(df)

st.subheader("Prediction")
species = np.array(["Adelie", "Gentoo", "Chinstrap"])
st.write(species[prediction][0])

st.subheader("Prediction Probability")
st.write(probability)
