import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay, precision_score,recall_score

df = pd.read_csv('breast-cancer.csv')
df.info()
df = df.drop('id',axis = 1)
def fill_diag(row):
    if row['diagnosis'] == 'M':
        return 0
    else:
        return 1
df['diagnosis'] = df.apply(fill_diag,axis= 1)
x = df.drop('diagnosis', axis= 1)
y = df['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#classifier = svm.SVC()
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
result = accuracy_score(y_test, y_pred)
resultx = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test,y_pred)
recl = recall_score(y_test,y_pred)
print(result)
print(resultx)
print(prec)
print(recl)
#LinearSVC
#0.986013986013986
#[[57  2] [ 0 84]]
#0.9767441860465116
#1.0
#SVC
#0.986013986013986
#[[52  1] [ 1 89]]
#0.9888888888888889
#0.9888888888888889
#KNN
#0.972027972027972
#[[52  3] [ 1 87]]
#0.9666666666666667
#0.9886363636363636
