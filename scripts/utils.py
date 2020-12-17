import datetime

import numpy as np
import pandas as pd

def type_converter(df):
    '''
    type 경량화 시키는 함수
    '''
    df.store_id = df.store_id.astype(np.int32)
    df.card_id = df.card_id.astype(np.int32)
    df.card_company = df.card_company.astype('category')
    df.transacted_date = pd.to_datetime(df.transacted_date + " " + df.transacted_time, format='%Y-%m-%d %H:%M')
    df.installment_term = df.installment_term.astype(np.int16)
    df.region = df.region.astype('category')
    df.type_of_business = df.type_of_business.astype('category')

    del df['transacted_time']
    return df


def submit_merger(prediction, f_name):
    '''
    prediction은 store_id와 amount 두 개 칼럼으로 이루어진 df
    '''
    submit = pd.read_csv("data09/submission.csv")
    submit = submit[['store_id']].merge(prediction, on='store_id', how='left')
    submit.amount = submit.amount.fillna(0)
    submit.to_csv(f'../data09/{f_name}.csv', index=False)
    
def train_test_splitter(df):
    '''
    주어진 데이터 중 마지막 3개월을 y로 둔다
    20%를 test로 분리
    '''
    y_bool = df.transacted_date >= datetime.datetime(2018, 12, 1)
    y = df[y_bool].groupby('store_id').amount.sum()
    X = df[~y_bool]
    
    train_index = y.sample(frac=0.8, random_state=85).index

    train_X = X[X.store_id.isin(train_index)]
    test_X = X[~X.store_id.isin(train_index)]

    train_y = y[y.index.isin(train_index)]
    test_y = y[~y.index.isin(train_index)]
    
    return train_X, test_X, train_y, test_y