import os
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score


# Load dirs name
cur_dir = os.path.realpath('.')
pos_dir = os.path.join(cur_dir, 'pos')
neg_dir = os.path.join(cur_dir, 'neg')

tfidfv = TfidfVectorizer(input='filename')

# Load files names
list_pos_dir = [ (os.path.join(pos_dir, x), 1) for x in os.listdir(pos_dir)][:2000]
list_neg_dir = [ (os.path.join(neg_dir, x), 0) for x in os.listdir(neg_dir)][:2000]
print("registers: {}".format(len(list_pos_dir+list_neg_dir)))
print("Attention with 6000 registers it will consume about 5+GB of ram")
input("Continue? or press CTRL+C")

# Mount data with label data frame
paths_df = pandas.DataFrame(list_pos_dir+list_neg_dir, columns=['path', 'label'])

# Load text and transform into vector
data = tfidfv.fit_transform(paths_df['path'].values)


#dataframe = pandas.DataFrame(data.toarray(), columns=tfidfv.get_feature_names())
#print(paths_df, "\n")
#print(paths_df['path'].head())
#print(list_pos_dir, list_neg_dir)
#print(tfidfv.get_feature_names(), len(tfidfv.get_feature_names()))
#print(tfidfv)
#print(dataframe.head())
#print(data, paths_df['label'].values, len(paths_df['label'].values))
#print(data.toarray())

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

a= []
for l in range(1,8):
    # Split into train and test
    X, x, Y, y = train_test_split(
        data.toarray(), paths_df.label, test_size=l/10.0, random_state=0
    )
    a.append(gnb())
    
    print(a)
df = pandas.DataFrame(a, columns=['Train', 'Test'],
        index=[x/10.0 for x in range(1,8)])
print(df)
from matplotlib import pyplot

pyplot.plot(df.Train)
pyplot.plot(df.Test)
pyplot.show()
