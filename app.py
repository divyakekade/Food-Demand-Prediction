from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd 
import csv
app = Flask(__name__)

model_weekly=pickle.load(open('./model_weekly.pkl','rb'))
model_monthly=pickle.load(open('./model_monthly.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/monthly-prediction',methods =["GET", "POST"])
def monthlyPredict():
    testing_result = [-1] 
    if request.method == "POST":
       month = int  (request.form.get("month"))
       food = int (request.form.get("food"))
       price = float (request.form.get("price"))
       try:  
            testing_values ={
            'price':[price],'Month':[month],'food_id':[food]
            }
            # print(testing_values['week'][0])
            df = pd.DataFrame.from_dict(testing_values)
            testing_result = model_monthly.predict(df)
            print(testing_result[0])
       except:
            print('error')
    
    if testing_result[0]==-1:
        return render_template("monthly-prediction.html")
    else:
        return render_template("monthly-prediction.html",prediction=testing_result[0])


@app.route('/weekly-prediction',methods =["GET", "POST"])
def weeklyPredict():
    testing_result = [-1]
    if request.method == "POST":
       week = int (request.form.get("week"))
       month = int  (request.form.get("month"))
       food = int (request.form.get("food"))
       price = float (request.form.get("price"))
       try:
            testing_values ={
            'week': [week],'price':[price],'Month':[month],'food_id':[food]
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
        return render_template("weekly-prediction.html",prediction=testing_result[0])


@app.route('/update', methods=["GET", "POST"])
def updateData():
    if request.method == "POST":
        data  = pd.read_csv('New_Data.csv')
        print(len(data))
        new_data = [7046, 3,234.4,3245,11,2]
        with open('New_Data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(new_data)
    return render_template("update.html")
  


if __name__ == '__main__':
    app.run(debug=True)
