# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 19:09:18 2020

@author: Shantanu
"""

import pandas as pd
import numpy as np

trainPath = './Data/train.csv'

trainData = pd.read_csv(trainPath)

#meanData = trainData["Number_Weeks_Used"].mean()

#meanDf = trainData["Number_Weeks_Used"].fillna(meanData)


trainDataFull = trainData.copy()

#trainDataFull["Number_Weeks_Used"] = meanDf

trainDataDroped = trainDataFull.dropna(subset=["Number_Weeks_Used"])

trainDataDroped = trainDataDroped.iloc[:, 1:]


dataForOHE = trainDataDroped[["Pesticide_Use_Category", "Season", "Crop_Type", "Soil_Type"]]
dataWithotCatData = trainDataDroped.drop(["Pesticide_Use_Category", "Season", "Crop_Type", "Soil_Type"], axis=1)

X = dataWithotCatData.iloc[:, :-1].values
y = trainDataDroped.iloc[:, -1].values


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
trainDataDropedScaled = scaler.fit_transform(X)


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
encData = enc.fit_transform(dataForOHE)
encDataArray = encData.toarray()

fullData = np.concatenate((trainDataDropedScaled, encDataArray), axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(fullData, y, test_size=0.2, random_state=42)



from sklearn.tree import DecisionTreeClassifier

svc = DecisionTreeClassifier(max_depth=3, criterion="entropy")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(score)

testPath = './Data/test.csv'
testData = pd.read_csv(testPath)

meanData = testData["Number_Weeks_Used"].mean()

means = testData["Number_Weeks_Used"].fillna(meanData)

testDataFull = testData.copy()

testDataFull["Number_Weeks_Used"] = means

testid = testDataFull.iloc[:, 0].values

DataWithoutId = testDataFull.iloc[:, 1:]

testDataforOHE = DataWithoutId[["Pesticide_Use_Category", "Season", "Crop_Type", "Soil_Type"]]
testdataWithotCatData = DataWithoutId.drop(["Pesticide_Use_Category", "Season", "Crop_Type", "Soil_Type"], axis=1)

testDataDropedScaled = scaler.transform(testdataWithotCatData.values)

enctestData = enc.transform(testDataforOHE)
enctestDataArray = enctestData.toarray()

fulltestData = np.concatenate((testDataDropedScaled, enctestDataArray), axis=1)

test_pred = svc.predict(fulltestData)

dataToSave = np.concatenate((testid.reshape(-1,1), test_pred.reshape(-1,1)), axis=1)
np.savetxt("pred.csv", dataToSave, delimiter=",", fmt="%s")