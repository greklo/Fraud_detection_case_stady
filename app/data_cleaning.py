import pandas as pd
import numpy as np

####### DATA CLEANING #######

def data_cleaning(df):
    '''
    Purpose: to clean and enrich data entered as a dataframe.
    
    Input: pandas dataframe consistent with a website's or other acquired JSON input. Usable for one doc / row or multiple rows.
    
    Output: pandas dataframe to be used for running predictive model or to create a predictive model
    '''
    
    # get count of number of previous payments
    df['num_previous_payouts'] = [len(i) for i in df['previous_payouts']]

    # convert to bool
    df['listed'] = [True if i=='y' else False for i in df['listed']]

    # TODO vectorize updates
    # misc changes
    df['num_previous_payouts'] = [len(i) for i in df['previous_payouts']]
    df['num_ticket_types'] = [len(i) for i in df['ticket_types']]
    df['description_length'] = [len(i) for i in df['description']]
    df['org_desc_length'] = [len(i) for i in df['org_desc']]
    df['listed'] = [True if i=='y' else False for i in df['listed']]
    # df['country_and_currency'] = [(i,df['currency'][idx]) for idx, i in enumerate(df['country'])]
    # df['country_and_currency'] = [str(i) + '_' + df['currency'][idx] for idx, i in enumerate(df['country'])] # TODO does not work with predict method
    df['payee_name_length'] = [len(i) for i in df['payee_name']]

    # todo bring back time variables back into model
    df['event_length'] = df['event_end'] - df['event_start']
    df['time_to_withdrawl'] = df['approx_payout_date'] - df['event_end']

    # TODO add country and currancy back in
    # create dummy variables
    df = pd.get_dummies(df, columns=['user_type', 'delivery_method', 'payout_type', ])#'country_and_currency'

    # convert time variables
    # time_cols = ['approx_payout_date', 'event_created', 'event_published','event_start', 'user_created', 'event_end']
    #
    # for col in time_cols:
    #     df[col] = df[col].astype('datetime64[s]')

    return df
