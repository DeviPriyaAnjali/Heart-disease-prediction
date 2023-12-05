# Heart-disease-prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing

data=pd.read_csv('/content/drive/MyDrive/Heart_Disease_Prediction.csv')

data

data.shape

data.info()

data.head()

data.tail()

data.tail()

data["Age"].value_counts()

data["BP"].value_counts()

print('The highest BP was of:',data['BP'].max())
print('The lowest BP was of:',data['BP'].min())
print('The average BP in the data:',data['BP'].mean())

print('The highest Cholesterol was of:',data['Cholesterol'].max())
print('The lowest cholesterol was of:',data['Cholesterol'].min())
print('The average cholesterol in the data:',data['Cholesterol'].mean())
import matplotlib.pyplot as plt
plt.plot(data['Cholesterol'])
plt.xlabel("Cholesterol")
plt.ylabel("Levels")
plt.title("Cholesterol plot")
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,5))
data_len=data[data['Heart Disease']=='Presence']['Cholesterol'].value_counts()

ax1.hist(data_len,color='red')
ax1.set_title('risk to heart disease')

data_len=data[data['Heart Disease']=='Absence']['Cholesterol'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('no risk to heart diseases')

fig.suptitle('cholesterol Levels')
plt.show()
data.duplicated()

data[data.duplicated()]

newdata=data.drop_duplicates()

newdata

data.isnull().sum()

data[1:5]

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression,LinearRegression

train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Heart Disease'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=data[data.columns[:-1]]
Y=data['Heart Disease']
len(train_X), len(train_Y), len(test_X), len(test_Y)

from sklearn.linear_model import LogisticRegression,LinearRegression
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
report = classification_report(test_Y, prediction3)
print("Classification Report:\n", report)

prediction3= model.predict(test_X)
conf_matrix =confusion_matrix(test_Y,prediction3)
print("confusion matrix :\n",conf_matrix)

from sklearn.metrics import precision_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
test_Y_encoded = label_encoder.fit_transform(test_Y)

predictions = model.predict(test_X)


predictions_encoded = label_encoder.transform(predictions)


precision = precision_score(test_Y_encoded, predictions_encoded)
accuracy = accuracy_score(test_Y_encoded, predictions_encoded)
f1 = f1_score(test_Y_encoded, predictions_encoded)


labels = ['Precision', 'Accuracy', 'F1 Score']
scores = [precision, accuracy, f1]

plt.bar(labels, scores, color=['blue', 'green', 'red'])
plt.title('Model Evaluation Metrics')
plt.ylabel('Score')
plt.show()

