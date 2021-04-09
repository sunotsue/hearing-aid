# Import Required Libraries

from flask import Flask,render_template,request,send_file,send_from_directory,jsonify
import numpy as np
import tensorflow 
from tensorflow import keras

# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name

app = Flask(__name__, template_folder='templates')

# We need to load the saved model file so as to use it for the prediction


nn = keras.models.load_model('concha_nn')


# For the root '/' we need to define a function in which we are rendering the template of index.html as default
# This rendering template is done if it get's any GET Request

@app.route('/',methods=['POST','GET'])
def main():
  if request.method=='GET':
    return render_template('index.html')

# For the root '/predict' we need to define a function named predict
# This function will take values from the ajax request and performs the prediction
# By getting response from flask to ajax it will display the response to the result field
# This whole above process occurs when request method is POST
# This rendering template is index.html if it get's any GET Request


@app.route('/predict',methods=['GET'])
def predict():
  #if request.method=='GET':
    #return render_template('index.html')
  if request.method == 'GET':
    # Converting all the form values to float and making them append in a list(features)
    features = [float(x) for x in request.form.values()]
    # Printing the features for debug purpose
    print(features)
    # Predicting the y values for the features collected
    pred = nn.predict([features])
    # Printing the y value array for debug purpose
    print(pred)
    return pred 
    
# It is the starting point of code
if __name__=='__main__':
  # We need to run the app to run the server
  app.run(debug=False) 

 # this threw an error because I had forgotten to flatten before passing off to dense model. Didn't have time to go back and modify my nn script in ipynb.  