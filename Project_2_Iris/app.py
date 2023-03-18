from flask import Flask, render_template, request, redirect
import pickle
import numpy as np

model= pickle.load(open("iris.pkl", "rb"))

app= Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods= ["POST"])
def predict_species():
    sl= float(request.form.get("sl"))
    sw= float(request.form.get("sw"))
    pl= float(request.form.get("pl"))
    pw= float(request.form.get("pw"))
    
    
    result= model.predict(np.array([[sl, sw, pl, pw]]))

    if result[0]== 0:
        return "<h1 style='color:green'>Setosa</h1>"

    elif result[0]== 1:
        return "<h1 style='color:red'>Versicolor</h1>"

    else:
        return "<h1 style='color:red'>Virginica</h1>" 

if __name__ == "__main__":
    app.run(host= "127.0.0.1", port= 5000, debug= True)