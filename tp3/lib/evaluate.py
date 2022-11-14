import numpy as np
import pandas as pd


def date_split(data: pd.DataFrame, cutYear=2016):
    train, test = data[data['release_date'] < cutYear], data[data['release_date'] >= cutYear]
    return train, test

def x_y_split(data: pd.DataFrame):
    cols = data.columns.to_list()
    cols.remove('genre')
    cols.remove('Unnamed: 0')
    return data[cols], data['genre']

def to_json_lines(X: pd.DataFrame, y: pd.Series, filename: str):
    f_x = open(f"X_{filename}.jl", 'w')
    f_y = open(f"y_{filename}.jl", 'w')
    X.to_json(f_x, orient='records', lines=True)
    y.to_json(f_y, orient='records', lines=True)
    f_x.close()
    f_y.close()


class Evaluator:
    def __init__(self, X_train, y_train, X_dev, y_dev, metric, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.evaluations = []

    def eval_pipe(self, model_name, pipe):
        res = self.eval_prediction(model_name, pipe.predict(self.X_train), pipe.predict(self.X_dev))
        if self.X_test is not None:
            res['test'] = self.metric(self.y_test, pipe.predict(self.X_test))
        return res

    def eval_prediction(self, model_name, y_hat_train, y_hat_dev):
        res = dict(
            name=model_name,
            train=self.metric(self.y_train, y_hat_train),
            dev=self.metric(self.y_dev, y_hat_dev)
        )

        self.evaluations.append(res)
        return res