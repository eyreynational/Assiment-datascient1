# TODO: Import Reuired Libraries
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import pandas as pd
data=pd.read_csv(filepath_or_buffer='./Advertising Budget and Sales.csv',
                 index_col=[0])
columns=["tv","radio","news","sales"]
data.columns=columns
dict_scaler=dict()
df = data.copy()
# st.write(df.head())
for col in columns:
    dict_scaler[col] = StandardScaler().fit(X=data[[col]])
target="sales"
features=data.columns.to_list()
features.remove(target)
X= data[features].values
y= data[target].values
robot=LinearRegression().fit(X=X, y=y)
st.title(body="Predict Sales by Advertisement Types")
user_input=dict()
for col in features:
     user_input[col]=st.number_input(label=col, value=data[col].mean())
#st.write(user_input)
df_input = pd.DataFrame(data=[user_input])
df_scaled=df_input.copy()
# st.write(features)
for col in features:
    df_scaled[col] = dict_scaler[col].transform(X=df_input[[col]])
X_test = df_scaled.values
st.dataframe(data=df_input)
# st.write(robot.predict(X=df_input.values))
y_pred = robot.predict(X=X_test)
y_pred = dict_scaler[target].inverse_transform(y_pred.reshape(-1,1))
st.write(y_pred)


