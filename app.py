import os
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle
import csv
app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

model_weekly=pickle.load(open('./weekly_model.pkl','rb'))
model_monthly=pickle.load(open('./monthly_model.pkl','rb'))

@app.route('/')
def home():
    Home_Image = os.path.join(app.config['UPLOAD_FOLDER'], 'food.jpg')
    return render_template("home.html", image=Home_Image)

@app.route('/weekly-section')
def weeklySection():
    return render_template("weekly-section.html")

@app.route('/monthly-section')
def monthlySection():
    return render_template("monthly-section.html")

@app.route('/monthly-prediction',methods =["GET", "POST"])
def monthlyPredict():
    testing_result = [-1] 
    if request.method == "POST":
       month = request.form.get("month")
       food = request.form.get("food")
       price = request.form.get("price")
       if(month=="" or food=="" or price==""):
            return render_template("monthly-prediction.html",error="Please fill values in all the fields.",month=month,price=price,food=food)
       try:  
            testing_values ={
            'price':[float(price)],'Month':[int(month)],'food_id':[int(food)]
            }
            df = pd.DataFrame.from_dict(testing_values)
            testing_result = model_monthly.predict(df)
            print(testing_result[0])
       except:
            print('error')
    
    if testing_result[0]==-1:
        return render_template("monthly-prediction.html")
    else:
        return render_template("monthly-prediction.html",prediction=int(testing_result[0]))


@app.route('/weekly-prediction',methods =["GET", "POST"])
def weeklyPredict():
    testing_result = [-1]
    if request.method == "POST":
       week = request.form.get("week")
       month = request.form.get("month")
       food = request.form.get("food")
       price = request.form.get("price")
       if(week=="" or month=="" or food=="" or price==""):
            return render_template("weekly-prediction.html",error="Please fill values in all the fields.")
       try:
            testing_values ={
            'week': [int(week)],'price':[float(price)],'Month':[int(month)],'food_id':[int(food)]
            }
            print(testing_values['week'][0])
            df = pd.DataFrame.from_dict(testing_values)
            testing_result = model_weekly.predict(df)
            print(testing_result[0])
       except:
            print('error')
    if testing_result[0]==-1:
        return render_template("weekly-prediction.html")
    else:
        return render_template("weekly-prediction.html",prediction=int(testing_result[0]))


@app.route('/weekly-update', methods=["GET", "POST"])
def weeklyUpdateData():
    if request.method == "POST":
        week = request.form.get("week")
        month = request.form.get("month")
        food = request.form.get("food")
        price = request.form.get("price")
        orders = request.form.get("orders")
        if(week=="" or month=="" or food=="" or price=="" or orders==""):
            return render_template("weekly-update.html",error="Please fill values in all the fields.")
        data  = pd.read_csv('weekly_train.csv')
        new_data = [len(data), int(week),float(price),int(orders),int(month),int(food)]
        with open('weekly_train.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_data)
        weeklydataUpdater()
        return render_template("weekly-update.html",success="Your data updated successfully!")
    return render_template("weekly-update.html")
  
@app.route('/monthly-update', methods=["GET", "POST"])
def monthlyUpdateData():
    if request.method == "POST":
        month = request.form.get("month")
        food = request.form.get("food")
        price = request.form.get("price")
        orders = request.form.get("orders")
        if(month=="" or food=="" or price=="" or orders==""):
            return render_template("monthly-update.html",error="Please fill values in all the fields.")
        data  = pd.read_csv('monthly_train.csv')
        new_data = [len(data),float(price),int(month),int(food),int(orders)]
        with open('monthly_train.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_data)
        monthlyDataUpdater()
        return render_template("monthly-update.html",success="Your data updated successfully!")
    return render_template("monthly-update.html")

def weeklydataUpdater():
    global model_weekly
    data = pd.read_csv('weekly_train.csv')
    # print(len(data))
    data = data.drop(['index'], axis=1)
    # X are features and Y is target column
    X = data.drop(columns='num_orders',axis=1)
    Y = data['num_orders']
    # Splitting the data into traing and testing data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    # Random forest algorithm
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, Y_train)
    training_data_predcition = rf.predict(X_train)
    r2_train = metrics.r2_score(Y_train, training_data_predcition)
    testing_data_predcition = rf.predict(X_test)
    r2_test = metrics.r2_score(Y_test, testing_data_predcition)
    pickle.dump(rf,open('weekly_model.pkl','wb'))
    model_weekly=pickle.load(open('weekly_model.pkl','rb'))

def monthlyDataUpdater():
    global model_monthly
    data = pd.read_csv('monthly_train.csv')
    data = data.drop(['index'], axis=1)
    # X are features and Y is target column
    X = data.drop(columns='num_orders',axis=1)
    Y = data['num_orders']
    # Splitting the data into traing and testing data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    """Random forest algorithm """

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, Y_train)
    pickle.dump(rf,open('monthly_model.pkl','wb'))
    model_monthly=pickle.load(open('monthly_model.pkl','rb'))

if __name__ == '__main__':
    app.run(debug=True)
