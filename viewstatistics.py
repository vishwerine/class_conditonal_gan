import matplotlib.pyplot as plt

f = open('results2/statistics.txt')

st = f.readline()

dloss = []
gloss = []

while st!="":
	dloss.append(st.split()[1])
	gloss.append(st.split()[3])
	st = f.readline()


dloss = [float(i) for i in dloss]
gloss = [float(i) for i in gloss]

dlossAvg = []
glossAvg = []

alpha = 0

step_size = 1000

for i in range(step_size,len(gloss),step_size):
	glossAvg.append(sum(gloss[alpha:i])/step_size)
	dlossAvg.append(sum(dloss[alpha:i])/step_size)
	alpha = i

plt.plot(glossAvg)
plt.plot(dlossAvg)

plt.show()

