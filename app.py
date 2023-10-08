
from flask import Flask,request,render_template,app
from flask import Response
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

scaler = pickle.load(open('Models/StandardScaler.pkl','rb'))
model = pickle.load(open('Models/LogisticRegression.pkl','rb'))

@app.route('/')
def homiee():
    return render_template('index.html')

@app.route('/predictdiabetes',methods = ['GET','POST'])
def predict_datapoint():
    result=''

    if request.method == 'POST' :

        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = int(request.form.get('Glucose'))
        BloodPressure = int(request.form.get('BloodPressure'))
        SkinThickness = int(request.form.get('SkinThickness'))
        Insulin = int(request.form.get('Insulin'))
        BMI = int(request.form.get('BMI'))
        DiabetesPedigreeFunction = int(request.form.get('DiabetesPedigreeFunction'))
        Age = int(request.form.get('Age'))

        new_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        prediction = model.predict(new_data)

        if prediction[0] == 1 :
            result = 'Diabetes'
        else :
            result = 'No Diabetes'

        return render_template('single_prediction.html',result=result)

    else :
        return render_template('home.html')

if __name__ == '__main__' :
    app.run(host='0.0.0.0')