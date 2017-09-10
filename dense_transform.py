from sklearn.base import TransformerMixin, BaseEstimator
import nltk

stemmer = nltk.stem.snowball.SnowballStemmer('english')

try:
    nltk.word_tokenize('')
except LookupError:
    nltk.download('punkt')

def tokenizer(text):
    tokens = nltk.word_tokenize(text)
    return [stemmer.stem(token) for token in tokens]
        
        


class DenseTransformer(TransformerMixin, BaseEstimator):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

