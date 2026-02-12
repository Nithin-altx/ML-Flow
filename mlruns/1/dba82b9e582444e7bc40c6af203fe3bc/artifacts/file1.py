import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# import seaborn as sns

wine=load_wine()
X = wine.data
y = wine.target


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=42)

max_depth=10
n_estimators=7

mlflow.set_experiment('Randomforest')

with mlflow.start_run():
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=42)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_artifact(__file__)


    print(accuracy)


