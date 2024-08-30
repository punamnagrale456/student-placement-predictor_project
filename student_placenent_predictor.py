import numpy as np
import pandas as pd


df = pd.read_csv(r'C:/Users/OMS-PUNE/Downloads/PUNAM_PROJECTS/students_placement.csv')
df

# df.shape


# X=df.drop(columns=['placed'])
# Y=df['placed']


# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# scaler = StandardScaler()
# X_train_trf = scaler.fit_transform(X_train)
# X_test_trf = scaler.transform(X_test)
# accuracy_score(y_test,
#                LogisticRegression()
#                .fit(X_train_trf,y_train)
#                .predict(X_test_trf))

# from sklearn.ensemble import RandomForestClassifier
# accuracy_score(y_test,RandomForestClassifier().fit(X_train,y_train).predict(X_test))

# from sklearn.svm import SVC
# accuracy_score(y_test,SVC(kernel='rbf').fit(X_train,y_train).predict(X_test))

# svc = SVC(kernel='rbf')
# svc.fit(X_train,y_train)

# import pickle 
# pickle.dump(svc,open('model.pkl','wb'))
