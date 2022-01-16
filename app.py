from flask import Flask, render_template, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the Model
model = pickle.load(open('model.pkl', 'rb'))

# Import scaler object used to fit the training data
from model_train import scaler

@app.route('/')
@app.route('/home')
@app.route('/predict')
def home():
    return render_template('index.html', Home='HOME')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = scaler.transform(np.array(float_features).reshape(1, -1))
    prediction = model.predict(final_features)
   
    if prediction:
        return render_template('index.html', positive_prediction_text='Patient is predicted to have Diabetes')
    else:
        return render_template('index.html', negative_prediction_text='Patient is predicted to NOT have Diabetes')
        
   

if __name__ == "__main__":
    app.run(debug=True)
