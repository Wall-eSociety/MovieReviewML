import os
import pandas
import numpy
from matplotlib import pyplot
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,  CountVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score


# Load dirs name
cur_dir = os.path.realpath('.')
pos_dir = os.path.join(cur_dir, 'pos')
neg_dir = os.path.join(cur_dir, 'neg')


# Load files names
list_pos_dir = [ (os.path.join(pos_dir, x), 1) for x in os.listdir(pos_dir)][:1000]
list_neg_dir = [ (os.path.join(neg_dir, x), 0) for x in os.listdir(neg_dir)][:1000]
print("registers: {}".format(len(list_pos_dir+list_neg_dir)))
print("Attention with 6000 registers it will consume about 5+GB of ram")
input("Continue? or press CTRL+C")

# Mount data with label data frame
paths_df = pandas.DataFrame(list_pos_dir+list_neg_dir, columns=['path', 'label'])

# Load text and transform into vector
cv = CountVectorizer(input='filename')
data = cv.fit_transform(paths_df['path'].values)



# gnb method
def gnb():
    global X, x, Y, y
    gnb = GaussianNB()
    # Complexy analysis
    #result = cross_val_score(gnb, data.toarray(), paths_df.label.values, cv=2)
    #print(result)
    # Simple analysis
    gnb.fit(X,Y)
    return (gnb.score(X, Y), gnb.score(x, y))

def gnb_binary():
    global X, x, Y, y
    gnb = BernoulliNB()
    # Complexy analysis
    #result = cross_val_score(gnb, data.toarray(), paths_df.label.values, cv=2)
    #print(result)
    # Simple analysis
    gnb.fit(X,Y)
    return (gnb.score(X, Y), gnb.score(x, y))

a= []
for l in range(1,8):
    # Split into train and test
    X, x, Y, y = train_test_split(
        data.toarray(), paths_df.label, test_size=l/10.0, random_state=0
    )
    a.append(gnb())

df = pandas.DataFrame(a, columns=['train_count', 'test_count'], index=[x/10.0 for x in range(1,8)])
#print(df.head())
b= []
data = TfidfTransformer().fit_transform(data.toarray())
for l in range(1,8):
    # Split into train and test
    X, x, Y, y = train_test_split(
        data.toarray(), paths_df.label, test_size=l/10.0, random_state=0
    )
    b.append(gnb())
b = numpy.array(b)
df = df.assign(train_tf=b[:,0], test_tf=b[:,1])
#print(df.head())

cv = CountVectorizer(input='filename', binary=True)
data = cv.fit_transform(paths_df['path'].values)

a= []
for l in range(1,8):
    # Split into train and test
    X, x, Y, y = train_test_split(
        data.toarray(), paths_df.label, test_size=l/10.0, random_state=0
    )
    a.append(gnb_binary())
a = numpy.array(a)
df = df.assign(train_count_binary=a[:,0], test_count_binary=a[:,1])
#print(df.head())

b = []
data = TfidfTransformer().fit_transform(data.toarray())
for l in range(1,8):
    # Split into train and test
    X, x, Y, y = train_test_split(
        data.toarray(), paths_df.label, test_size=l/10.0, random_state=0
    )
    b.append(gnb_binary())
b = numpy.array(b)
df = df.assign(train_tf_binary=b[:,0], test_tf_binary=b[:,1])
#print(df.head())

pyplot.subplot(221)
pyplot.title('Count vectorization')
pyplot.plot(df.train_count)
pyplot.plot(df.test_count)
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")

pyplot.subplot(222)
pyplot.title('Term frequency inverse document-frenquency')
pyplot.plot(df.train_tf)
pyplot.plot(df.test_tf)
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")


pyplot.subplot(223)
pyplot.title('Count vectorization binary')
pyplot.plot(df.train_count_binary)
pyplot.plot(df.test_count_binary)
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")

pyplot.subplot(224)
pyplot.title('Term frequency inverse document-frenquency binary')
pyplot.plot(df.train_tf_binary)
pyplot.plot(df.test_tf_binary)
pyplot.ylabel("Score")
pyplot.xlabel("% size train base")
pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

pyplot.show()
print(df.ix[df.idxmax()])
