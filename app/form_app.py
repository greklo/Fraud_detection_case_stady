from flask import Flask, render_template, request, jsonify
import pickle
try:
    from app.model_create import rft_model, get_data

except:
    from model_create import rft_model, get_data
import urllib.request, json
import pandas as pd

app = Flask(__name__)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def submit():
    """Render a page with a 'start' button to grab a json file from a hard-coded
       Galvanize associated website. """
    return render_template('form/submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    with urllib.request.urlopen("http://galvanize-case-study-on-fraud.herokuapp.com/data_point") as url:
        new_data = json.loads(url.read().decode())

    print(new_data)
    new_data = pd.DataFrame.from_dict(new_data, orient = 'index').T
    pred = model.predict(new_data)

    new_data['prediction'] = pred
    new_data = new_data.rename(index=str, columns={"object_id": "_id"}).to_dict('records')[0]

    return render_template('form/predict.html', _id=new_data['_id'], predicted=pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
