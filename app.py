from flask import Flask, render_template, request, jsonify, redirect
import joblib

app = Flask(__name__)


@app.before_request
def before_request():
    if not request.is_secure:
        url = request.url.replace('http://', 'https://', 1)
        code = 301
        return redirect(url, code=code)


def diabetes_predict(algo, preg, glucose, bp, st, insulin, bmi, dpf, age):
    """
    It takes in the name of the algorithm, and the features of the patient, and returns the prediction
    of the model
    
    :param algo: The algorithm to use for prediction
    :param preg: Number of times pregnant
    :param glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    :param bp: Blood pressure
    :param st: Skin Thickness
    :param insulin: 2-Hour serum insulin (mu U/ml)
    :param bmi: Body mass index
    :param dpf: Diabetes Pedigree Function
    :param age: Age (years)
    :return: The prediction of the model.
    """
    data = [[preg, glucose, bp, st, insulin, bmi, dpf, age]]
    print(data)
    if algo == "knn":
        model = joblib.load("models/knn_diabetes.pkl")
        prediction = model.predict(data)
        return int(prediction[0])
    elif algo == "naive":
        model = joblib.load("models/nb_diabetes.pkl")
        prediction = model.predict(data)
        return int(prediction[0])
    else:
        return "Please select a valid algorithm"


@app.route('/')
def knn():
    """
    It renders the home.html file.
    :return: The render_template function is being returned.
    """
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    It takes the values from the form and passes them to the diabetes_predict function
    :return: a jsonfield dictionary.
    """
    if request.method == "POST":
        # get body parameters
        algo = request.json['algo']
        preg = int(request.json['preg'])
        glucose = int(request.json['glucose'])
        bp = int(request.json['bp'])
        st = int(request.json['st'])
        insulin = int(request.json['insulin'])
        bmi = float(request.json['bmi'])
        dpf = float(request.json['dpf'])
        age = int(request.json['age'])

        try:
            prediction = diabetes_predict(algo, preg, glucose, bp, st, insulin, bmi, dpf, age)

            return jsonify({
                "prediction": prediction
            })
        except ValueError:
            return jsonify({
                "error": "Please enter valid values"
            })


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
