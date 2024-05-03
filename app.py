import streamlit as st
import pandas as pd
import joblib
from sklearn import preprocessing

model = joblib.load('random_forest_model.pkl')

def preprocess_input(input_data):
    df = input_data
    label_encoder = preprocessing.LabelEncoder()
    df['Sex']= label_encoder.fit_transform(df['Sex'])
    df['Cabin']= label_encoder.fit_transform(df['Cabin'])
    df['Embarked']= label_encoder.fit_transform(df['Embarked'])
    return df

def predict_survival(input_data):  
    input_data_processed = preprocess_input(input_data)
    prediction = model.predict(input_data_processed)
    return prediction

st.title('Titanic Survival Prediction')

pclass = st.selectbox('Passenger Class (1st, 2nd, 3rd)', [1, 2, 3])
sex = st.radio('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0, max_value=1000, value=50)
cabin = st.text_input('Cabin', '')
embarked = st.selectbox('Embarked (C, Q, S)', ['C', 'Q', 'S'])

input_data = {
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Cabin': cabin,
    'Embarked': embarked
}

if st.button('Predict Survival'):
    input_df = pd.DataFrame([input_data])
    prediction = predict_survival(input_df)
    if prediction[0] == 1:
        st.markdown('<h2 style="color: green;">Survived</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="color: red;">Did not survive</h2>', unsafe_allow_html=True)