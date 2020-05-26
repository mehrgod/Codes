import numpy as np
def coinflip(p):
	return 1 if np.random.rand()<=p else 0

def A0():
	return np.random.choice(3)

def A1():
	return np.random.randint(20, 41)

def A2(c):
	if c == 0: return coinflip(.8)
	if c == 1: return coinflip(.1)

def A3():
	return coinflip(.5)

def A4():
	return np.random.randint(8)

def A5(c):
	max = 5 + c*5
	return np.random.randint(max)

def A6():
	return np.random.randint(100,120)/100.

def A7(c):
	min = 1 if c == 0 else 5
	max = 8 if c == 0 else 10
	return np.random.randint(min, max)

def A8(c):
	min = 1 if c == 0 else 3
	max = 7 if c == 0 else 10
	return np.random.randint(min, max)

def A9(c):
	return np.random.poisson(2 + 2*c)

def datum(c):
	return [A0(), A1(), A2(c), A3(), A4(), A5(c), A6(), A7(c), A8(c), A9(c), c]

N_train = 100
p0 = .7
p1 = 1-p0

N_test = 10

data = []
test = []

for n in range(int(N_train*p0)):
	data.append(datum(0))
for n in range(int(N_train*p1)):
	data.append(datum(1))

for n in range(int(N_test*p0)):
	test.append(datum(0))
for n in range(int(N_test*p1)):
	test.append(datum(1))	

data = np.array(data)
np.random.shuffle(data)

test = np.array(test)
np.random.shuffle(test)

# np.savetxt("train.txt",data, fmt=['%d','%d','%d','%d','%d','%d','%0.2f','%d','%d','%d','%d'],delimiter=",")
np.savetxt("test.txt",test, fmt=['%d','%d','%d','%d','%d','%d','%0.2f','%d','%d','%d','%d'],delimiter=",")
