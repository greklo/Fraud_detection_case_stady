import pandas as pd
import numpy as np
import pickle
from pymongo import MongoClient
from app.model_create import rft_model

def predict(new_data):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)


    #test_data = pd.read_json('files/data.json').sample(1)

    new_data['prediction'] = model.predict(new_data)

    new_data = new_data.rename(index=str, columns={"object_id": "_id"}).to_dict('records')[0]

    return new_data, new_data['_id']

    #

    # client = MongoClient()
    # db = client.fraud_case_study
    #
    # collection = db.test_collection
    #
    # collection.insert_one(new_data)
    # print(collection.find_one())

if __name__ == '__main__':
    predict()


##### mongo instructions
# mongo
# use fraud_case_study
# db.test_collection.find()
