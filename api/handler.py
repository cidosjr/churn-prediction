import pickle
from flask import Flask, request, Response
import pandas as pd
from churn.Churn import Churn


# load model
path = '/home/cid/repos/churn-prediction/src/models/'
model = pickle.load( open(path + 'model_rf.pkl', 'rb') )

app = Flask(__name__)

# endpoint
@app.route('/churn/predict', methods=['POST'])
def churn_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        pipeline = Churn()
        
        # data preprocessing
        df1 = pipeline.data_preprocessing(test_raw)
        
        # prediction 
        df_response = pipeline.get_prediction(model, test_raw, df1)
        
        return df_response       
        
        
    else:
        return Response('{}', status=200, mimetype='aplication/json')
    

if __name__ == '__main__':
    app.run('0.0.0.0')
