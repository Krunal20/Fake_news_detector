from flask import Flask, request, jsonify
from waitress import serve 
import joblib
import pandas as pd
import traceback
import sys

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    if model1 and model2 :
        try :
            query = pd.DataFrame(request.json)
            query['xdata'] = query['title'] + " " + query['text']
            query = query.drop(['title', 'text'], axis = 1)
            query = query.reindex(columns = model_columns, fill_value = "nothing")
            query = tf.transform(query['xdata'])
            prediction1 = list(model1.predict(query))
            prediction2 = list(model2.predict(query))
            for i in range(len(prediction1)) :
                if prediction1[i] == 1 :
                    prediction1[i] = 'Real'
                else :
                    prediction1[i] = 'Fake'
            for i in range(len(prediction2)) :
                if prediction2[i] == 1 :
                    prediction2[i] = 'Real'
                else :
                    prediction2[i] = 'Fake'
            return jsonify({"prediction1" : prediction1, "prediction2" : prediction2})
        except :
            return jsonify({"trace": traceback.format_exc()})
    else :
        print("Train the model first")
        return("No model found!")

if __name__ == "__main__":
    try:
        port = int(sys.argv[2]) # This is for a command-line input
        host = str(sys.argv[1])
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
        host = '0.0.0.0'

    tf = joblib.load("tfidfvectorizer.pkl") # Load vectorizer
    model1 = joblib.load("model1.pkl") # Load "model1.pkl"
    model2 = joblib.load("model2.pkl") # Load "model2.pkl"
    model_columns = joblib.load("cols.pkl") # Load "model_columns.pkl"

    serve(app=app, host=host, port=port)       
    