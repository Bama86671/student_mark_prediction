from copyreg import pickle
from pyexpat import model
import numpy as np
import pickle
from flask import render_template , request , Flask

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('Marks.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    m1 = request.form.get('mark1')
    m2 = request.form.get('mark2')
    noOfAbsense = request.form.get('absense')
    print(f"Data = {noOfAbsense}, {m1}, {m2}")

    features = np.array([int(noOfAbsense), int(m1), int(m2)]).reshape(1, -1)
    prediction = model.predict(features)
    print(prediction)
    return render_template('Marks.html',predict = f' You can get {prediction}% Marks')


app.run(debug=True)