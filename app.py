from flask import Flask, render_template, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
# scaler = StandardScaler()
from model_train import scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = scaler.transform(np.array(float_features).reshape(1, -1)) #[np.array(int_features)]
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    if prediction:
        return render_template('index.html', prediction_text='Patient is predicted to have Diabetes')
    else:
        return render_template('index.html', prediction_text='Patient is predicted to NOT have Diabetes')
        
    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(port=5000, debug=True)
