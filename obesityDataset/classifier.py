import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm
#from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay, precision_score,recall_score

df = pd.read_csv('obesity.csv')
df.info()
df = df.drop('ID',axis= 1)
def fill_gender(row):
    if row['Gender'] == 'Male':
        return 0
    else:
        return 1
df['Gender'] = df.apply(fill_gender,axis= 1)
