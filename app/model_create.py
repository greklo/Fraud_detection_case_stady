import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
from data_cleaning import data_cleaning

class rft_model():
    def __init__(self,keep_cols,df):
        '''
        Flexible object to take in various columns - cleaned by data_cleaning.py - to create a predictive model and transform it to a pickle file/object.
        
        Input: 
        1) the columns used to create the model in list format
        2) the dataframe of existing data
        
        Output: Pickle model, Random Forest Tree Classifier Object
        
        '''
        self.keep_cols = keep_cols # features for model
        self._tts(df)              # Create test_train split
        self.create_model()        # run model with random forest classifier
        self.scores()              # run scoring for the model
        
    def _tts(self,df):
        '''
        Break out data into train test split data groups for use in different types of models. Generalized our model building object.
        
        Input: Pandas DataFrame of data to be split
        Output: X_train,y_train,X_test,y_test (X's are Pandas DataFrames with relevent data, Y's are lists of the target values
        '''
        y = df['frauds']
        X = df[self.keep_cols]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y)

    def create_model(self):
        '''
        Create a model based on training information. Used 250 trees as perfected during EDA.
        
        Input: None. Run while initialization
        Output: None. Model object is trained and created.
        
        '''
        self.rfc = RandomForestClassifier(n_estimators=250)
        self.rfc.fit(self.X_train, self.y_train)
  
    def scores(self):
        '''
        Scoring the model based on supervised target data, and using test and training data.
        Creates a confusion matrix, recall, precision, accuracy which is stored in the object.
        
        input: None. Activated in init function.
        
        Output: None. Stores and prints out the confusion matrix, Precision, Recall, Accuracy.
        
        '''
        # self.test_score = self.rfc.score(self.X_test, self.y_test)
        # self.train_score = self.rfc.score(self.X_train, self.y_train)

        self.y_predict = self.rfc.predict(self.X_test)
        conmatrix = confusion_matrix(self.y_test, self.y_predict)
        tn, fp, fn, tp = conmatrix.ravel() # turns confusion matrix into a 1D array

        # create performance metrics
        self.precision = tp / float(tp + fp)
        self.recall = tp / float(tp + fn)
        self.accuracy = (tp + tn) / float(tp + tn +fp + fn)

        # show results
        print(conmatrix)
        print('Precision is: ', self.precision)
        print('Recall is: ', self.recall)
        print('Accuracy is: ', self.accuracy)

    def predict(self, new_data):
        '''
        Object method called by the Flask predict function to declare True/False for Fraud detection
        
        Input: Pandas DataFrame of grabbed JSON information
        
        Output: True / False prediction for the row of event information
        
        '''
        new_data = get_data(new_data)

        # TODO check to see if original is bool
        for col in self.X_train.columns.tolist():
            if col in new_data.columns.tolist():
                continue

            new_data[col] = 0

        return self.rfc.predict(new_data[self.keep_cols])

def get_data(data):
    
    '''
    Initial data cleaning code required for model building.
    
    Input: Pandas DataFrame containing information to train on
    
    Output: Pandas DataFrame containing cleaned information to train on
    '''
    df = data

    # Identifying target for model
    if 'acct_type' in df.columns:
        frauds = ['fraudster_event', 'fraudster', 'fraudster_att']
        df['frauds'] = [True if i in frauds else False for i in df['acct_type']]

    return data_cleaning(df)


if __name__ == '__main__':

    # Import data. Do general data cleaning
    # TODO reference data_cleaning.py file

    df = pd.read_json('files/data.json')
    df = get_data(df)

    # # input parameters here
    # # creates test train split & creates model

    # TODO move keep cols to separate file. OR dynamically create based on dummy base
    keep_cols = [u'event_length', u'time_to_withdrawl', u'fb_published', u'sale_duration2', 'num_previous_payouts', 'payee_name_length', u'gts', 'num_ticket_types', u'num_order', u'channels', 'description_length', 'org_desc_length', u'num_payouts', u'has_logo', u'show_map', u'listed', u'has_analytics', u'name_length', 'user_type_1', 'user_type_2', 'user_type_3', 'user_type_4', 'user_type_5', 'user_type_103', 'delivery_method_0.0', 'delivery_method_1.0', 'delivery_method_3.0', u'payout_type_', u'payout_type_ACH', u'payout_type_CHECK']

    # keep_cols = ['num_previous_payouts']
    rft = rft_model(df = df, keep_cols = keep_cols)
    #
    # # show performance of model
    rft.scores()
    #
    # # Pickle model for use later
    with open('model2.pkl', 'wb') as f:
        pickle.dump(rft, f)

