import os
from pickle import load

import numpy as np
from flask import Flask, request, render_template
from distutils.util import strtobool

app = Flask(__name__)
scaler = load(open(os.path.join('model', 'scaler.pkl'), 'rb'))
model = load(open(os.path.join('model', 'RFR_AirbnbModel.model'), 'rb'))


# function to predict the price
def prediction(X_test):
    x = np.asarray([X_test]).astype(np.float32)
    log_price = list(model.predict(scaler.transform(x)).flatten())
    price = list(np.exp(log_price).flatten())
    return log_price[0], price[0]


@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        formData = request.form.to_dict()
        formData['cleaning_fee'] = strtobool(formData['cleaning_fee'])
        log_price, price = prediction(list(map(int, list(formData.values()))))
        kwargs = {'price': "The price of the selected property is {:.2f}$".format(price)}
        return render_template('index.html', **kwargs)


if __name__ == '__main__':
    app.run(port=8080)
