import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle

data = pd.read_csv('New_Data.csv')
# data = data.drop(['index'], axis=1)
# X are features and Y is target column
X = data.drop(columns='num_orders',axis=1)
Y = data['num_orders']
# Splitting the data into traing and testing data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

"""Random forest algorithm """

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

training_data_predcition = rf.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_predcition)

testing_data_predcition = rf.predict(X_test)
r2_test = metrics.r2_score(Y_test, testing_data_predcition)
testing_values ={
            'week': [1],'price':[231],'Month':[1],'food_id':[1]
            }
# print(testing_values['week'][0])
# df = pd.DataFrame.from_dict(testing_values)
# testing_result = rf.predict(df)
# print(testing_result[0])
pickle.dump(rf,open('weekly_model.pkl','wb'))
model=pickle.load(open('weekly_model.pkl','rb'))


