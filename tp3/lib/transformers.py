from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureProjection(BaseEstimator, TransformerMixin):
    """
    Recibe una lista de campos a proyectar, y los proyecta como listas o como diccionarios
    Ver notebook 02
    """

    def __init__(self, fields, as_dict=False, convert_na=True):
        self.fields = fields
        self.as_dict = as_dict
        self.convert_na = convert_na

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = []
        for a in X.iterrows():
            doc = a[1]
            if self.as_dict:
                row = {field: doc[field] for field in self.fields}
            else:
                row = [doc[field] for field in self.fields]
            res.append(row)
        return res


class FeatureProjectionFromJson(BaseEstimator, TransformerMixin):
    """
    Recibe una lista de campos a proyectar, y los proyecta como listas o como diccionarios
    Ver notebook 02
    """

    def __init__(self, fields, as_dict=False, convert_na=True):
        self.fields = fields
        self.as_dict = as_dict
        self.convert_na = convert_na

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = []
        for i, doc in enumerate(X):
            if self.as_dict:
                row = {field: doc[field] for field in self.fields}
            else:
                row = [doc[field] for field in self.fields]
            res.append(row)
        return res


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Encodea una categorica como un vector de cuatro dimensiones
    [mean(y), std(y), percentile(y, 5), percentile(y, 95)]
    """

    def __init__(self, categorical_field, min_freq=5):
        self.min_freq = min_freq
        self.categorical_field = categorical_field
        self.stats_ = None
        self.default_stats = None

    def fit(self, X, y):
        values = defaultdict(list)
        for i, x in enumerate(X):
            values[x[self.categorical_field]].append(y[i])

        self.stats_ = {}
        for cat_value, tar_values in values.items():
            if len(tar_values) < self.min_freq: continue
            tar_values = np.asarray(tar_values)
            freq = np.bincount(tar_values)
            self.stats_[cat_value] = list(np.argsort(freq)[::-1][:2])


        freq = np.bincount(y)
        self.default_stats_ = list(np.argsort(freq)[::-1][:2])
        # Siempre hay que devolver self
        return self

    def transform(self, X):
        res = []
        for i, doc in enumerate(X):
            vector = self.stats_.get(doc[self.categorical_field], self.default_stats_)
            res.append(vector)
        return res
    
    
class TargetEncodeWithNumericalMean(BaseEstimator, TransformerMixin):
    """
    Encodea una categorica como un vector con el promedio de las numericas que les pases
    [mean(numerica1), mean(numerica2), mean(numerica3), ...]
    """

    def __init__(self, categorical_field, numerical_fields, min_freq=5):
        self.min_freq = min_freq
        self.categorical_field = categorical_field
        self.numerical_fields = numerical_fields
        self.stats_ = None
        self.default_stats = None

    def fit(self, X, y):
        values = defaultdict(list)
        self.stats_ = {}
        num_field_dict = defaultdict(list)
        for i, x in enumerate(X):
            values[x[self.categorical_field]].append(x) #Separo las rows por categoria
        
        for cat_value, tar_values in values.items():
            if len(tar_values) < self.min_freq: continue #Si la categoria tiene mas de min_freq rows
            cat_value_vector = []
            for num_field in self.numerical_fields: #Hago el promedio por categoria de cada variable numerica
                num_field_vector = []
                for tar_value in tar_values:
                    num_field_vector.append(tar_value[num_field])
                  
                num_field_vector = np.asarray(num_field_vector)
                num_field_dict[num_field].append(np.mean(num_field_vector)) #Guardo el promedio numerico de la categoria
                cat_value_vector.append(np.mean(num_field_vector))
                         
            self.stats_[cat_value] = cat_value_vector
                
            
        
        self.default_stats_ = [np.mean(np.asarray(num_values)) for num_values in list(num_field_dict.values())]
        
   
        # Siempre hay que devolver self
        return self

    
    
    def transform(self, X):
        res = []
        for i, doc in enumerate(X):
            vector = self.stats_.get(doc[self.categorical_field], self.default_stats_)
            res.append(vector)
        return res

class TargetEncodeWithNumericalMaxOutlier(BaseEstimator, TransformerMixin):
    """
    Encodea una categorica como un vector con el promedio de las numericas que les pases
    [mean(numerica1), mean(numerica2), mean(numerica3), ...]
    """

    def __init__(self, categorical_field, numerical_fields, min_freq=5):
        self.min_freq = min_freq
        self.categorical_field = categorical_field
        self.numerical_fields = numerical_fields
        self.stats_ = None
        self.default_stats = None

    def fit(self, X, y):
        values = defaultdict(list)
        self.stats_ = {}
        num_field_dict = defaultdict(list)
        for i, x in enumerate(X):
            values[x[self.categorical_field]].append(x) #Separo las rows por categoria
        
        for cat_value, tar_values in values.items():
            if len(tar_values) < self.min_freq: continue #Si la categoria tiene mas de min_freq rows
            cat_value_vector = []
            for num_field in self.numerical_fields: #Hago el promedio por categoria de cada variable numerica
                num_field_vector = []
                for tar_value in tar_values:
                    num_field_vector.append(tar_value[num_field])
                  
                num_field_vector = np.asarray(num_field_vector)
                num_field_dict[num_field].append(np.percentile(num_field_vector,90)) #Guardo el promedio numerico de la categoria
                cat_value_vector.append((np.percentile(num_field_vector,90)))
                         
            self.stats_[cat_value] = cat_value_vector
                
            
        
        self.default_stats_ = [np.percentile(np.asarray(num_values),90) for num_values in list(num_field_dict.values())]
        
   
        # Siempre hay que devolver self
        return self

    
    
    def transform(self, X):
        res = []
        for i, doc in enumerate(X):
            vector = self.stats_.get(doc[self.categorical_field], self.default_stats_)
            res.append(vector)
        return res

class TargetEncodeWithNumericalMinOutlier(BaseEstimator, TransformerMixin):
    """
    Encodea una categorica como un vector con el promedio de las numericas que les pases
    [mean(numerica1), mean(numerica2), mean(numerica3), ...]
    """

    def __init__(self, categorical_field, numerical_fields, min_freq=5):
        self.min_freq = min_freq
        self.categorical_field = categorical_field
        self.numerical_fields = numerical_fields
        self.stats_ = None
        self.default_stats = None

    def fit(self, X, y):
        values = defaultdict(list)
        self.stats_ = {}
        num_field_dict = defaultdict(list)
        for i, x in enumerate(X):
            values[x[self.categorical_field]].append(x) #Separo las rows por categoria
        
        for cat_value, tar_values in values.items():
            if len(tar_values) < self.min_freq: continue #Si la categoria tiene mas de min_freq rows
            cat_value_vector = []
            for num_field in self.numerical_fields: #Hago el promedio por categoria de cada variable numerica
                num_field_vector = []
                for tar_value in tar_values:
                    num_field_vector.append(tar_value[num_field])
                  
                num_field_vector = np.asarray(num_field_vector)
                num_field_dict[num_field].append(np.percentile(num_field_vector,10)) #Guardo el promedio numerico de la categoria
                cat_value_vector.append((np.percentile(num_field_vector,10)))
                         
            self.stats_[cat_value] = cat_value_vector
                
            
        
        self.default_stats_ = [np.percentile(np.asarray(num_values),10) for num_values in list(num_field_dict.values())]
        
   
        # Siempre hay que devolver self
        return self

    
    
    def transform(self, X):
        res = []
        for i, doc in enumerate(X):
            vector = self.stats_.get(doc[self.categorical_field], self.default_stats_)
            res.append(vector)
        return res

class PretrainedFastTextTransformer(BaseEstimator, TransformerMixin):
    """
    Dado un nombre de archivo de un modelo de fasttext (ver notebook 4a y 4b) y un campo de texto
    Genera features del campo textual a traves del modelo de fasttext
    """

    def __init__(self, fname, field):
        self.fname = fname
        self.field = field
        self.model_ = None

    def sync_resources(self):
        if self.model_ is None:
            # Lazy import. Solo falla si lo usas.
            try:
                import fasttext
            except ImportError:
                raise ImportError('Falta instalar fasttext. \n \n pip install fasttext \n \n')
            self.model_ = fasttext.load_model(self.fname)

    def fit(self, X, y):
        self.sync_resources()
        return self

    def transform(self, X):
        self.sync_resources()
        res = []
        for doc in X:
            value = doc[self.field].replace('\n', '')
            res.append(self.model_.get_sentence_vector(value))
        return np.asarray(res)

    def __getstate__(self):
        state = vars(self).copy()
        state['model_'] = None
        return state

    def __setstate__(self, state):
        vars(self).update(state)

