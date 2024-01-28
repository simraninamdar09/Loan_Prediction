#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Loan prediction.csv")

#Replacing +3 to3
data['Dependents'] = data['Dependents'].replace('3+','3')
#skewness of column
#data['Loan_Amount_Term'] = np.log(data['Loan_Amount_Term'])
#dropping column
data.drop(columns=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_ID'],inplace= True)

#filling null values
data['Gender'].fillna(data['Gender'].mode()[0],inplace = True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace = True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace = True)
data['Married'].fillna(data['Married'].mode()[0],inplace = True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace = True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace = True)

# Encoding data
col2 = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
#encode data
label_encoder = preprocessing.LabelEncoder()
for col in col2:
    data[col] =  label_encoder.fit_transform(data[col])

# Split the data into features (X) and target (y)
x = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Bagging
bag_c=BaggingClassifier()
bag1 =bag_c.fit(x_train,y_train)
bag1.score(x_train,y_train)
bag1.score(x_test,y_test)
#Pickel file
filename = 'final_Bagging_model.pkl'
pickle.dump(bag_c, open(filename, 'wb'))
bag1.fit(x, y)
pk = bag1.predict(x_test)


Gender = st.selectbox('Gender', data['Gender'].unique())
Married = st.selectbox('Married', data['Married'].unique())
Dependents = st.selectbox('Dependents', data['Dependents'].unique())
Education = st.selectbox('Education', data['Education'].unique())
Self_Employed = st.selectbox('Self_Employed', data['Self_Employed'].unique())
Loan_Amount_Term = st.selectbox('Loan_Amount_Term', data['Loan_Amount_Term'].unique())
Credit_History = st.selectbox('Credit_History', data['Credit_History'].unique())
Property_Area = st.selectbox('Property_Area', data['Property_Area'].unique())

if st.button('Prevention Type'):
    df = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History' : Credit_History,
        'Property_Area' : Property_Area
    }

    df1 = pd.DataFrame(df, index=[1])
    pred = bag1.predict(df1)

    if bag1.predict(pred) == 1:
         print("Loan approved")
    else:
       print("Not Loan Approved")
    st.title("Loan Status " + str(prediction_value))








