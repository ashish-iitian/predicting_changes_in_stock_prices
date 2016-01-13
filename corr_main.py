import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from corr_feature_extractor import feature_extractor
from corr_predictor import regression
from sklearn import cross_validation


def train(lag, y_train, train_set):

	'''  Training phase: set of features taken, processed duly for lag_param, fed to linear regression 
		module for learning the model. Model learnt is validated through cross-validation. Returns 
		model and features for test phase to the main '''

	extractor = feature_extractor(lag)
	feature_matrix = extractor.processor(train_set)
	model = regression.learner(feature_matrix[:50], y_train)
#	print cross_validator(model, feature_matrix[:50], y_train)
	return model, feature_matrix[50:]


''' --------------------------- for cross validation -------------------------------- '''
def cross_validator(regr, features, Y):

	''' does cross validation (here 3-fold) for trained model and returns scores based on r2 metric '''

	kfold = cross_validation.cross_val_score(regr, features, Y, cv = 3, scoring='r2')
	return kfold	


def test(model, test_data):

	''' Test phase: set of features for test set fed to the learned model for predicting unknowns '''

	return regression.predictor(model, test_data)


def main():

	''' handles the function calls and writes predicted values to a csv file for output '''

	df = pd.read_csv(r'./data/stock_returns_base150.csv',skipfooter = 50)
	dates = df.ix[: , ['date']].iloc[50:].reset_index()
	dates.drop('index', axis=1, inplace=True)
	label = df.ix[:49,['S1']].astype(float).values.tolist()
	data = df.ix[:, ['S2','S3','S4','S5','S6','S7','S8','S9','S10']]
	for column in data:
		df[column] = df[column].astype(float)
	length = len(data)
	train_data = data.values.tolist()
	lag_param = 0
	model_learned, test_features = train(lag_param, label, train_data)
	y_pred = test(model_learned, test_features).flatten()
	value = pd.DataFrame({'Value':y_pred[:]}, index=range(len(y_pred)))
	result = pd.concat([dates,value],axis=1)
	result.to_csv('predictions.csv', index=False)


if __name__ == "__main__":
	main()
