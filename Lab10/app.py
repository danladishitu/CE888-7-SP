#This is Heroku Deployment Lecture
from flask import Flask, request, render_template
import os
import pickle

print("Test")
print("Test 2")
print(os.getcwd())
path = os.getcwd()

with open('Models/Pickle_LR_Model.pkl', 'rb') as f:
    logistic = pickle.load(f)

with open('Models/Pickle_RM_Model.pkl', 'rb') as f:
    randomforest = pickle.load(f)

with open('Models/Pickle_SVM_Model.pkl', 'rb') as f:
    svm_model = pickle.load(f)


def get_predictions(age, sex, cp, trestbps, chol, fbs, restecg, req_model, thalach, exang, oldpeak, slope,ca, thal):
    mylist = [age, sex, cp, trestbps, chol, fbs,restecg,req_model, thalach, exang, oldpeak, slope, ca, thal]
    mylist = [float(i) for i in mylist]
    vals = [mylist]

    if req_model == 'Logistic':
        #print(req_model)
        return logistic.predict(vals)[0]

    elif req_model == 'RandomForest':
        #print(req_model)
        return randomforest.predict(vals)[0]

    elif req_model == 'SVM':
        #print(req_model)
        return svm_model.predict(vals)[0]
    else:
        return "Cannot Predict"


app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        req_model = request.form['req_model']
        thalach= request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']

        target = get_predictions(age, sex, cp, trestbps, chol, fbs, restecg,req_model,
                    thalach, exang, oldpeak, slope,ca,thal)

        if target==1:
            sale_making = 'Patient is likely to have an heart disease'
        else:
            sale_making = 'Patient is unlikely to have an heart disease'

        return render_template('home.html', target = target, sale_making = sale_making)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)