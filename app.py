import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder(handle_unknown='ignore')
app=Flask(__name__)
model= pickle.load(open("/Users/KIIT/LinearRegressionModel.pkl",'rb'))
car = pd.read_csv('/Users/KIIT/cleaned car project.csv')

@app.route('/', methods=['GET','POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0,"select company")
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    return str(prediction[0])
if __name__ == '__main__':
    app.run(debug=True)
