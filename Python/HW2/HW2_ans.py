import numpy as np

def doSplit(D, index, value):
	"""Splits a dataset according to the split rule given
	Not a required function

	D - dataset, tuple (X, y) where X is data, y is classes
	Args:
		index: index of attribute to split on
		value: splits x<=value, x>value
	Returns:
		a tuple containing counts in each partition
		a tuple (size 2) containing lists of probabilities for each class in each partition
		a list of classes in the data
	"""
	X = D[0]
	y = D[1]

	classes = [c for c in set(y)]
	#The below is fine for a small number of classes; for very many indices, avoid dicts
	partitioned_counts = {decision:{c:0 for c in classes} for decision in range(2)}

	for i in range(len(X)):
		decision = 0 if X[i][index]<=value else 1
		partitioned_counts[decision][y[i]]+=1

	counts = tuple([np.sum([partitioned_counts[d][c] for c in classes]) for d in range(2)])
	probs = tuple([[(float(partitioned_counts[d][c])/counts[d]) if counts[d] > 0 else 0 for c in classes] for d in range(2)])

	return counts, probs, classes

def IG(D, index, value):
	"""Compute the Information Gain of a split on attribute index at value
	for dataset D.
	
	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the Information Gain for the given split
	"""
	
	counts, probs, classes = doSplit(D, index, value)
	N = float(np.sum(counts))
	y = D[1].tolist()
	probs_orig = [y.count(c)/N for c in set(y)]

	HD = -2*np.sum([p*np.log2(p) for p in probs_orig])
	HDy = -2*np.sum([p*np.log2(p) for p in probs[0] if p > 0])
	HDn = -2*np.sum([p*np.log2(p) for p in probs[1] if p > 0])

	IG = HD - counts[0]/N*HDy - counts[1]/N*HDn

	return IG

def G(D, index, value):
	"""Compute the Gini index of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the Gini index for the given split
	"""

	counts, probs, classes = doSplit(D, index, value)
	N = float(np.sum(counts))

	GDy = 1-np.sum([p**2 for p in probs[0]])
	GDn = 1-np.sum([p**2 for p in probs[1]])

	Gini = counts[0]/N*GDy + counts[1]/N*GDn

	return Gini

def CART(D, index, value):
	"""Compute the CART measure of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the CART measure for the given split
	"""

	counts, probs, classes = doSplit(D, index, value)
	N = float(np.sum(counts))

	CART = 2* (counts[0]/N)* (counts[1]/N) * np.sum([abs(probs[0][i] - probs[1][i]) for i in range(len(probs[0]))])

	return CART

def bestSplit(D, criterion):
	"""Computes the best split for dataset D using the specified criterion

	Args:
		D: A dataset, tuple (X, y) where X is the data, y the classes
		criterion: one of "IG", "GINI", "CART"

	Returns:
		A tuple (i, value) where i is the index of the attribute to split at value
	"""

	#functions are first class objects in python, so let's refer to our desired criterion by a single name
	if criterion == "IG": criterion_function=IG
	if criterion == "GINI": criterion_function=G
	if criterion == "CART": criterion_function=CART

	X = D[0]
	num_attr = X.shape[1]
	split_scores = {}
	for index in range(num_attr):
		vals = sorted(set(X[:,index]))
		for value in vals:
			split_scores[(index,value)] = criterion_function(D, index, value)

	if criterion == "IG" or criterion == "CART": 
		return max(split_scores, key=split_scores.get)
	else: 
		return min(split_scores, key=split_scores.get)

def load(filename):
	"""Loads filename as a dataset. Assumes the last column is classes, and 
	observations are organized as rows.

	Args:
		filename: file to read

	Returns:
		A tuple D=(X,y), where X represents data points and y their classes
	"""
	all_data = np.loadtxt(filename, delimiter=",")
	X = all_data[:,:-1]
	y = all_data[:,-1]

	return (X, y)

def classifyIG(train, test):
	"""Builds a single-split decision tree using the Information Gain criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		pred: a list of predicted classes for observations in test (in order)
	"""
	(index, value) = bestSplit(train, "IG")
	counts, probs, classes = doSplit(train, index, value)
	classifier = [np.argmax(probs[0]), np.argmax(probs[1])]


	pred = []
	for obs in test[0]:
		decision = 0 if obs[index] <= value else 1
		pred.append(classifier[decision])

	return pred

def classifyG(train, test):
	"""Builds a single-split decision tree using the GINI criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		pred: a list of predicted classes for observations in test (in order)
	"""
	(index, value) = bestSplit(train, "GINI")
	counts, probs, classes = doSplit(train, index, value)
	classifier = [np.argmax(probs[0]), np.argmax(probs[1])]


	pred = []
	for obs in test[0]:
		decision = 0 if obs[index] <= value else 1
		pred.append(classifier[decision])

	return pred

def classifyCART(train, test):
	"""Builds a single-split decision tree using the CART criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		pred: a list of predicted classes for observations in test (in order)
	"""
	(index, value) = bestSplit(train, "CART")
	counts, probs, classes = doSplit(train, index, value)
	classifier = [np.argmax(probs[0]), np.argmax(probs[1])]


	pred = []
	for obs in test[0]:
		decision = 0 if obs[index] <= value else 1
		pred.append(classifier[decision])

	return pred


def main():
	"""This portion of the program will run when run only when main() is called.
	This is good practice in python, which doesn't have a general entry point 
	unlike C, Java, etc. 
	This way, when you <import HW2>, no code is run - only the functions you
	explicitly call.
	"""
	train = load("train.txt")
	test = load("test.txt")
	bi = bestSplit(train, "IG")
	print bi, IG(train, bi[0], bi[1])
	bg = bestSplit(train, "GINI")
	print bg, G(train, bg[0], bg[1])
	bc = bestSplit(train, "CART")
	print bc, CART(train, 7, 2)
	pred_IG = classifyIG(train, test)
	pred_G = classifyG(train, test)
	pred_CART = classifyCART(train, test)

	y = test[1]
	print np.sum(abs(y-pred_IG))
	print np.sum(abs(y-pred_G))
	print np.sum(abs(y-pred_CART))


if __name__=="__main__": 
	"""__name__=="__main__" when the python script is run directly, not when it 
	is imported. When this program is run from the command line (or an IDE), the 
	following will happen; if you <import HW2>, nothing happens unless you call
	a function.
	"""
	main()