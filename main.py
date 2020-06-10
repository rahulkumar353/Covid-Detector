from flask import Flask, render_template, request
import pickle

main = Flask(__name__)

pickle_in = open('corona.pkl', 'rb')
model = pickle.load(pickle_in)


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        fever = request.form['fever']
        breath = request.form['breath']
        cold = request.form['cold']
        pain = request.form['pain']

        input = [[int(age), int(fever), int(cold), int(breath), int(pain)]]

        predict = model.predict(input)[0]
        predict_proba = model.predict_proba(input)[0][1]

        return render_template('result.html', predict=predict, proba=round(predict_proba * 100, 2))

    return render_template('index.html')


if __name__ == '__main__':
    main.run(debug=True)
