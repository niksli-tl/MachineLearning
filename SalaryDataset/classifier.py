import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay, precision_score,recall_score

df = pd.read_csv('salary.csv')
df = df.drop(['education','fnlwgt'],axis= 1)
def fill_workclass(row):
    if row['workclass'] == ' Local-gov':
        return 1
    elif row['workclass'] == ' State-gov':
        return 1
    elif row['workclass'] == ' Federal-gov':
        return 1
    else:
        return 0
df['workclass'] = df.apply(fill_workclass,axis= 1)
def fill_race(row):
    if row['race'] == ' White':
        return 0
    elif row['race'] == ' Black':
        return 1
    else:
        return 0
df['race'] = df.apply(fill_race,axis= 1)
def fill_sex(row):
    if row['sex'] == ' Male':
        return 0
    else:
        return 1
df['sex'] = df.apply(fill_sex,axis= 1)
def fill_nc(row):
    if row['native-country'] == ' United-States':
        return 1
    else:
        return 0
df['native-country'] = df.apply(fill_nc,axis= 1)
def fill_salary(row):
    if row['salary'] == ' <=50K':
        return 0
    else:
        return 1
df['salary'] = df.apply(fill_salary,axis= 1)
df[list(pd.get_dummies(df['marital-status']).columns)] = pd.get_dummies(df['marital-status'])
df[list(pd.get_dummies(df['occupation']).columns)] = pd.get_dummies(df['occupation'])
df[list(pd.get_dummies(df['relationship']).columns)] = pd.get_dummies(df['relationship'])
df = df.drop(['marital-status','occupation','relationship'],axis= 1)
x = df.drop('salary', axis= 1)
y = df['salary']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = svm.LinearSVC()
#classifier = KNeighborsClassifier(n_neighbors=9)
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
#KNN
#0.8318388404372927
#[[5600  537] [ 832 1172]]
#0.6857811585722645
#0.5848303393213573
#SVC
#0.8452278589853827
#[[5777  412] [ 848 1104]]
#0.7282321899736148
#0.5655737704918032
#LinearSVC
#0.8438766736273186
#[[5700  435] [ 836 1170]]
#0.7289719626168224
#0.5832502492522432
