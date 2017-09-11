import os
import pandas
import numpy
from dense_transform import DenseTransformer, tokenizer
from matplotlib import pyplot
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,  CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# Load dirs name
cur_dir = os.path.realpath('.')
pos_dir = os.path.join(cur_dir, 'pos')
neg_dir = os.path.join(cur_dir, 'neg')


# Load files names
list_pos_dir = [ (os.path.join(pos_dir, x), 1) for x in os.listdir(pos_dir)][:50]
list_neg_dir = [ (os.path.join(neg_dir, x), 0) for x in os.listdir(neg_dir)][:50]
print("registers: {}".format(len(list_pos_dir+list_neg_dir)))
print("Attention with 6000 registers it will consume about 5+GB of ram")
# input("Continue? or press CTRL+C")

# Mount data with label data frame

paths_df = pandas.DataFrame(list_pos_dir+list_neg_dir, columns=['path', 'label'])

# Verify difference between size of tokens with tokenizer stem, stopwords
tfidf_stem = TfidfVectorizer(input='filename', stop_words='english', tokenizer=tokenizer)
tfidf_stop = TfidfVectorizer(input='filename', stop_words='english')
tfidf_word = TfidfVectorizer(input='filename')

result = []

for tfidf in [tfidf_stem, tfidf_word, tfidf_stop]:
    tfidf.fit(paths_df.path.values)
    result.append(len(tfidf.get_feature_names()))

result = pandas.DataFrame(result, columns=['len_of_features'], index=['tfidf_stem', 'tfidf_word', 'tfidf_stop'])
result = result.assign(difference=lambda x: (x.len_of_features - x.len_of_features.min()))
print(result)
pyplot.figure(3)
pyplot.bar([1,2,3], result.difference.values)
pyplot.xticks([1,2,3], result.index.values)
pyplot.ylabel('Number of tokens')
pyplot.xlabel('Method of tf-idf')

pipes = {
    'gaussianNB': Pipeline([
      ('vect', TfidfVectorizer(input='filename')),
      ('dense', DenseTransformer()),
      ('gnb', GaussianNB())
    ]),
    'bernoulliNB': Pipeline([
      ('vect', TfidfVectorizer(input='filename', binary=True)),
      ('dense', DenseTransformer()),
      ('gnb', BernoulliNB())
    ]),
    'multinomialNB': Pipeline([
      ('vect', TfidfVectorizer(input='filename')),
      ( 'gnb', MultinomialNB())
    ]),
    'linearSVC': Pipeline([
      ('vect', TfidfVectorizer(input='filename')),
      ( 'gnb', LinearSVC())
    ]),
    'sgdclassifier': Pipeline([
      ('vect', TfidfVectorizer(input='filename')),
      ( 'gnb', SGDClassifier(max_iter=5))
    ]),
}

def extract_params(best_params_):
    return {'ngram_range': best_params_['vect__ngram_range'],
      'use_idf': best_params_['vect__use_idf'],
      'norm':  best_params_['vect__norm'],
      'sublinear_tf': best_params_['vect__sublinear_tf'],
      'stop_words': best_params_['vect__stop_words'],
      'tokenizer': best_params_['vect__tokenizer']
    }

parameters = {
  'vect__ngram_range': [(1,1), (1,2)],
  'vect__use_idf': (True, False),
  'vect__norm':  ('l2', 'l1', None),
  'vect__sublinear_tf': (True, False),
  'vect__stop_words': ('english', None),
  'vect__tokenizer': (None, tokenizer),
}

parametrized = GridSearchCV(pipes['gaussianNB'], parameters, n_jobs=3)
parametrized.fit(paths_df.path,paths_df.label)
print(parametrized.best_score_, parametrized.best_params_)

pipes['optimizedgaussianNB'] = Pipeline([
      ('vect', TfidfVectorizer(input='filename', **extract_params(parametrized.best_params_))),
      ('dense', DenseTransformer()),
      ('gnb', GaussianNB())
    ])

parametrized = GridSearchCV(pipes['bernoulliNB'], parameters, n_jobs=3)
parametrized.fit(paths_df.path,paths_df.label)
print(parametrized.best_score_, parametrized.best_params_)

pipes['optimizedbernoulliNB'] = Pipeline([
      ('vect', TfidfVectorizer(input='filename', binary=True, **extract_params(parametrized.best_params_))),
      ('dense', DenseTransformer()),
      ('gnb', BernoulliNB())
    ])

parametrized = GridSearchCV(pipes['multinomialNB'], parameters, n_jobs=3)
parametrized.fit(paths_df.path,paths_df.label)
print(parametrized.best_score_, parametrized.best_params_)

pipes['optimizedmultinomialNB'] = Pipeline([
      ('vect', TfidfVectorizer(input='filename', **extract_params(parametrized.best_params_))),
      ( 'gnb', MultinomialNB())
    ])

# parametrized = GridSearchCV(pipes['linearSVC'], parameters, n_jobs=3)
# parametrized.fit(paths_df.path,paths_df.label)
# print(parametrized.best_score_, parametrized.best_params_)
# 
# pipes['optimizedlinearSVC'] = Pipeline([
#       ('vect', TfidfVectorizer(input='filename', **extract_params(parametrized.best_params_))),
#       ('gnb', LinearSVC())
#     ])
# 
# parametrized = GridSearchCV(pipes['sgdclassifier'], parameters, n_jobs=3)
# parametrized.fit(paths_df.path,paths_df.label)
# print(parametrized.best_score_, parametrized.best_params_)
# 
# pipes['optimizedsgdclassifier'] = Pipeline([
#       ('vect', TfidfVectorizer(input='filename', **extract_params(parametrized.best_params_))),
#       ('gnb', SGDClassifier())
#     ])

index = [x/10.0 for x in range(1,8)]

df = pandas.DataFrame(index=index)
for pipe_name, pipe in pipes.items():
    temp = []
    for l in range(1,8):
        # Split into train and test
        # X - Train, Y - Train
        # x - test, y - test
        X, x, Y, y = train_test_split(
            paths_df.path, paths_df.label, test_size=l/10.0, random_state=0
        )
        pipe.fit(X,Y)
        temp.append([pipe.score(X,Y), pipe.score(x,y)])
    columns = ["train_{}".format(pipe_name), "test_{}".format(pipe_name)]
    new_df = pandas.DataFrame(temp, columns=columns, index=index)
    df = df.join(new_df)

pyplot.figure(1)


pyplot.subplot(221)
pyplot.title('Gaussian NB')
pyplot.plot(df.train_gaussianNB)
pyplot.plot(df.test_gaussianNB)
pyplot.plot(df.test_optimizedgaussianNB, 'r--')
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")

pyplot.subplot(222)
pyplot.title('BernoulliNB')
pyplot.plot(df.train_bernoulliNB)
pyplot.plot(df.test_bernoulliNB)
pyplot.plot(df.test_optimizedbernoulliNB, 'r--')
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")


pyplot.subplot(223)
pyplot.title('MultinomialNB')
pyplot.plot(df.train_multinomialNB)
pyplot.plot(df.test_multinomialNB)
pyplot.plot(df.test_optimizedmultinomialNB, 'r--')
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")

pyplot.figure(4)
pyplot.subplot(222)
pyplot.title('LinearSVC')
pyplot.plot(df.train_linearSVC)
pyplot.plot(df.test_linearSVC)
# pyplot.plot(df.test_optimizedlinearSVC, 'r--')
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")


pyplot.subplot(221)
pyplot.title('SGDClassifier')
pyplot.plot(df.train_sgdclassifier)
pyplot.plot(df.test_sgdclassifier)
# pyplot.plot(df.test_optimizedsgdclassifier, 'r--')
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")
pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
print(df.ix[df.idxmax()])

remove_stopword = Pipeline([
      ('vect', TfidfVectorizer(input='filename', stop_words='english')),
      ('dense', DenseTransformer()),
      ('gnb', GaussianNB())
])

temp = []
for l in range(1,8):
    X, x, Y, y = train_test_split(
        paths_df.path, paths_df.label, test_size=l/10.0, random_state=0
    )
    remove_stopword.fit(X,Y)
    temp.append([remove_stopword.score(X,Y), remove_stopword.score(x,y)])
stop_words = numpy.array(temp)

pyplot.figure(2)

pyplot.subplot(211)
pyplot.title('With stop words')
pyplot.plot(df.train_gaussianNB)
pyplot.plot(df.test_gaussianNB)
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")
pyplot.subplot(212)
pyplot.title('Without stop words')
pyplot.plot(stop_words[:,0])
pyplot.plot(stop_words[:,1])
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")

pyplot.show()
# #######TRASH CODE##################

# Load text and transform into vector
# cv = CountVectorizer(input='filename')
# data = cv.fit_transform(paths_df['path'].values)
# 
# 
# 
# # gnb method
# def gnb():
#     global X, x, Y, y
#     gnb = GaussianNB()
#     # Complexy analysis
#     #result = cross_val_score(gnb, data.toarray(), paths_df.label.values, cv=2)
#     #print(result)
#     # Simple analysis
#     gnb.fit(X,Y)
#     return (gnb.score(X, Y), gnb.score(x, y))
# 
# def gnb_binary():
#     global X, x, Y, y
#     gnb = BernoulliNB()
#     # Complexy analysis
#     #result = cross_val_score(gnb, data.toarray(), paths_df.label.values, cv=2)
#     #print(result)
#     # Simple analysis
#     gnb.fit(X,Y)
#     return (gnb.score(X, Y), gnb.score(x, y))
# 
# a= []
# for l in range(1,8):
#     # Split into train and test
#     X, x, Y, y = train_test_split(
#         data.toarray(), paths_df.label, test_size=l/10.0, random_state=0
#     )
#     a.append(gnb())
# 
# df = pandas.DataFrame(a, columns=['train_count', 'test_count'], index=[x/10.0 for x in range(1,8)])
# #print(df.head())
# b= []
# data = TfidfTransformer().fit_transform(data.toarray())
# for l in range(1,8):
#     # Split into train and test
#     X, x, Y, y = train_test_split(
#         data.toarray(), paths_df.label, test_size=l/10.0, random_state=0
#     )
#     b.append(gnb())
# b = numpy.array(b)
# df = df.assign(train_tf=b[:,0], test_tf=b[:,1])
# #print(df.head())
# 
# cv = CountVectorizer(input='filename', binary=True)
# data = cv.fit_transform(paths_df['path'].values)
# 
# a= []
# for l in range(1,8):
#     # Split into train and test
#     X, x, Y, y = train_test_split(
#         data.toarray(), paths_df.label, test_size=l/10.0, random_state=0
#     )
#     a.append(gnb_binary())
# a = numpy.array(a)
# df = df.assign(train_count_binary=a[:,0], test_count_binary=a[:,1])
# #print(df.head())
# 
# b = []
# data = TfidfTransformer().fit_transform(data.toarray())
# for l in range(1,8):
#     # Split into train and test
#     X, x, Y, y = train_test_split(
#         data.toarray(), paths_df.label, test_size=l/10.0, random_state=0
#     )
#     b.append(gnb_binary())
# b = numpy.array(b)
# df = df.assign(train_tf_binary=b[:,0], test_tf_binary=b[:,1])
# #print(df.head())
# 
# pyplot.subplot(221)
# pyplot.title('Count vectorization')
# pyplot.plot(df.train_count)
# pyplot.plot(df.test_count)
# pyplot.ylabel("Score")
# pyplot.xlabel("% size train base")
# 
# pyplot.subplot(222)
# pyplot.title('Term frequency inverse document-frenquency')
# pyplot.plot(df.train_tf)
# pyplot.plot(df.test_tf)
# pyplot.ylabel("Score")
# pyplot.xlabel("% size train base")
# 
# 
# pyplot.subplot(223)
# pyplot.title('Count vectorization binary')
# pyplot.plot(df.train_count_binary)
# pyplot.plot(df.test_count_binary)
# pyplot.ylabel("Score")
# pyplot.xlabel("% size train base")
# 
# pyplot.subplot(224)
# pyplot.title('Term frequency inverse document-frenquency binary')
# pyplot.plot(df.train_tf_binary)
# pyplot.plot(df.test_tf_binary)
# pyplot.ylabel("Score")
# pyplot.xlabel("% size train base")
# pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
#                     wspace=0.35)
# 
# pyplot.show()
# print(df.ix[df.idxmax()])
