from sklearn.base import TransformerMixin, BaseEstimator
import nltk
import re
from textblob import TextBlob

stemmer = nltk.stem.snowball.SnowballStemmer('english')

try:
    nltk.word_tokenize('')
except LookupError:
    nltk.download('punkt')

def tokenizer(text):
    tokens = nltk.word_tokenize(text)
    return [stemmer.stem(token) for token in tokens]
        
def alphanumeric_tokens(data):
    tokens = re.sub('[^a-zA-Z]', ' ', data)
    return tokens
        
def tokenizer_alphanumeric(text):
    return tokenizer(new_text)
    
def tokenizer_correct(text):
    tokens = TextBlob(text)
    tokens = tokens.correct()
    return tokens.words


class DenseTransformer(TransformerMixin, BaseEstimator):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

