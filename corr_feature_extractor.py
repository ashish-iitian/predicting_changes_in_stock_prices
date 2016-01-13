import numpy as np
from itertools import chain


class feature_extractor:

	def __init__(self, lag):
		self.lag = lag

	def pad_for_lag_param(self, feature_set):

		''' applies padding so that the treatment for lag_param is consistent for the first set of features.
			For eg, for lag_param = 2, we will replicate the first entry of feature_set twice with itself.
			Say, feature_set = [A, B, C] where A, Band C are list of features for 3 different days.
			We will pad above so that it becomes [A, A, A, B, C], thus first day having (A, A, A) as 
			features, 2nd day having (B, A, A) and 3rd day having (C, B, A) as features for lag_param=2 '''

		pad = [feature_set[0] for x in xrange(self.lag)]
		return pad + feature_set
	
	def processor(self, feature_file):

		''' processes features matrix passed by train() to implement lag_param and returns post-processed
			features file to train() '''

		list_features = []
		feature_file = self.pad_for_lag_param(feature_file)
		length = len(feature_file)
		list_features = [list(chain.from_iterable(feature_file[i:i+1+self.lag])) for i in xrange(length-self.lag)]
		return list_features

