from sklearn.linear_model import Ridge	#LinearRegression

class regression:

	@staticmethod
	def learner(feat_matrix, y_train):

		''' trains a linear regression model to fit to the training data set and returns learned model '''

		linReg = Ridge(alpha=1)	#LinearRegression()
		return linReg.fit(feat_matrix, y_train)

	@staticmethod
	def predictor(model, test_features):

		''' runs the model on test data to return predictions by our linear regression model '''

		return model.predict(test_features)
