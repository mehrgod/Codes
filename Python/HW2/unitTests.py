import HW2_ans as ans
import HW2_ as test #Change to import different HW2.py I guess?
import numpy as np

def testLoad():
	passed = 0
	if (ans.load("train.txt")[0][34] == np.array(test.load("train.txt")[0][34])).all(): 
		passed += 1
	else:
		print("failed load train.txt\n")

	if (ans.load("test.txt")[1] == np.array(test.load("test.txt")[1])).all():
		passed += 1
	else:
		print("failed load test.txt\n")

	if (ans.load("train2.txt")[0][10] == np.array(test.load("train2.txt")[0][10])).all():
		passed += 1
	else:
		print("failed load train2\n")

	if(ans.load("test2.txt")[1] == np.array(test.load("test2.txt")[1])).all():
		passed +=1
	else:
		print("failed load test2\n")

	print("load() passed "+str(passed)+"/4\n")

def testCriterion(criterion):
	if criterion == "IG": 
		af = ans.IG
		tf = test.IG
	if criterion == "GINI": 
		af = ans.G
		tf = test.G
	if criterion == "CART": 
		af = ans.CART
		tf = test.CART

	passed = 0

	aD = ans.load("train.txt")
	tD = test.load("train.txt") #just in case their data loading doesn't match ours...

	if abs(af(aD, 3, 0) - tf(tD, 3, 0)) < .001:
		passed +=1
	else:
		print("failed "+criterion+" i3, v0\n")

	if abs(af(aD, 7, 4) - tf(tD, 7, 4)) < .001:
		passed +=1
	else:
		print("failed "+criterion+" i7, v4\n")

	if abs(af(aD, 5, 4) - tf(tD, 5, 4)) < .001:
		passed +=1
	else:
		print("failed "+criterion+" i5, v4\n")

	if abs(af(aD, 2, 0) -tf(tD, 2, 0)) < .001:
		passed +=1
	else:
		print("failed "+criterion+" i2, v0\n")

	if abs(af(aD, 9, 3) - tf(tD, 9, 3)) < .001:
		passed +=1
	else:
		print("failed "+criterion+" i9, v3\n")

	print(criterion+" passed "+str(passed)+"/5\n")

def testBestSplit():
	passed = 0
	aD = ans.load("train.txt")
	tD = test.load("train.txt") #just in case their data loading doesn't match ours...
    
	if (ans.bestSplit(aD, "IG") == test.bestSplit(tD, "IG")):
		passed +=1 
	else:
		print("failed bestSplit IG\n")
    
	if (ans.bestSplit(aD, "GINI") == test.bestSplit(tD, "GINI")):
		passed +=1 
	else:
		print("failed bestSplit GINI\n")

	if (ans.bestSplit(aD, "CART") == test.bestSplit(tD, "CART")):
		passed +=1 
	else:
		print("failed bestSplit CART\n")

	aD2 = ans.load("train2.txt")
	tD2 = test.load("train2.txt")
	if (ans.bestSplit(aD2, "IG") == test.bestSplit(tD2, "IG")):
		passed +=1 
	else:
		print("failed bestSplit IG 2\n")

	if (ans.bestSplit(aD2, "GINI") == test.bestSplit(tD2, "GINI")):
		passed +=1 
	else:
		print("failed bestSplit GINI 2\n")

	if (ans.bestSplit(aD2, "CART") == test.bestSplit(tD2, "CART")):
		passed +=1 
	else:
		print("failed bestSplit CART 2\n")

	print("passed bestSplit "+str(passed)+"/6\n")

def testClassify(criterion):
	if criterion == "IG": 
		af = ans.classifyIG
		tf = test.classifyIG
	if criterion == "GINI": 
		af = ans.classifyG
		tf = test.classifyG
	if criterion == "CART": 
		af = ans.classifyCART
		tf = test.classifyCART

	aD = ans.load("train.txt")
	aT = ans.load("test.txt")

	tD = test.load("train.txt")
	tT = test.load("test.txt")

	passed = 0

	if (af(aD, aT) == tf(tD, tT)):
		passed += 1
	else:
		print("failed classify "+criterion+"\n")

	aD2 = ans.load("train2.txt")
	aT2 = ans.load("test2.txt")

	tD2 = test.load("train2.txt")
	tT2 = test.load("test2.txt")

	if (af(aD2, aT2) == tf(tD2, tT2)):
		passed += 1
	else:
		print("failed classify 2 "+criterion+"\n")

	print("passed classify "+criterion+" "+str(passed)+"/2\n")

def main():
	testLoad()
	testCriterion("IG")
	testCriterion("GINI")
	testCriterion("CART")
	testBestSplit()
	testClassify("IG")
	testClassify("GINI")
	testClassify("CART")

if __name__=="__main__":
	main()




