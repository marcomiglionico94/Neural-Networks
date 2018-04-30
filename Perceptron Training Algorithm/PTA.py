import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Class Pta, it can be easily reused to perform the different experiments required in the Homework
class Pta:

# Initialise the parameters
    def __init__(self, omega, omega_2, mis, s, row):
        self.mis = mis
        self.misclass = []
        self.num = 0
        self.epochNum = []
        self.omega = omega
        self.omega_2 = omega_2
        self.omegaFinal = omega_2
        self.s = s
        self.row = row

# Running PTA algorithm
    def pta_alg(self, desiredOutput, actualOutput, eta):
        self.epochNum= np.append(self.epochNum,0)
        self.misclass = np.append(self.misclass, self.mis)

        while self.mis != 0:  # While the number of misclassifications is not equal to 0(convergence)
            output = []
            for i in range(0, self.row):
                dot = np.dot(self.s[i, :], self.omegaFinal)  # dot product
                if dot >= 0:
                    output.append(1)
                else:
                    output.append(0)

                self.omegaFinal = self.omegaFinal + eta * self.s[i, :] * (desiredOutput[i] - output[i]) # omega update

            self.num = self.num + 1
            # Append the number of the current epoch
            self.epochNum = np.append(self.epochNum, self.num)
            # Find the new number of misclassifications
            self.mis = np.count_nonzero(output != desiredOutput)
            # Put in the vector the number of misclassifications found for this epoch
            self.misclass = np.append(self.misclass, self.mis)
            print("Total number of misclassifications after " + str(self.num) + " epoch= " + str(self.mis))


    def get_omega_final(self):
        return self.omegaFinal

    def get_num(self):
        return self.num

    def get_misclass(self):
        return self.misclass

    def get_epoch(self):
        return self.epochNum


def create_grid(p, left, right, step1, step2):
    # Set grid parameters
    p.grid(which='both')
    major_ticks = np.arange(left, right, step1)
    minor_ticks = np.arange(left, right, step2)
    p.set_xticks(major_ticks)
    p.set_xticks(minor_ticks, minor=True)
    p.set_yticks(major_ticks)
    p.set_yticks(minor_ticks, minor=True)
    p.grid(which='minor', alpha=0.3)
    p.grid(which='major', alpha=0.7)


# PICK THE RANDOM WEIGHTS
print("WEIGHTS")
# Pick w0 uniformly at random
w0 = np.random.uniform(-0.25, 0.25)
# Pick w1 uniformly at random
w1 = np.random.uniform(-1, 1)
# Pick w1 uniformly at random
w2 = np.random.uniform(-1, 1)

# Parameters needed to create the x vectors
column = 2  # elements in each vector
row = 1000  # number of vectors created

# Create random vectors x independently and uniformly at random
values = np.random.uniform(-1, 1, row*column)
s = np.array(values).reshape(row, column)

# Insert the bias(1) in the vectors
s = np.insert(s, 0, 1, axis=1)

# Create omega,the vector of weights
omega = np.asarray([w0, w1, w2])
print("VECTOR OF WEIGHTS: " + str(omega))

tmp1 = []
tmp2 = []
# Output will be a vector containing only 0(indicates class S0) and 1(indicate class S1)
output = []

# Dot product
for i in range(0, row):
    dot = np.dot(s[i, :], omega)
    if dot >= 0:
        tmp1.append(s[i, 1:3])
        output.append(1)
    else:
        tmp2.append(s[i, 1:3])
        output.append(0)

# Divide in the two set S1 and S0
s1 = np.array(tmp1)
s0 = np.array(tmp2)
# Vector of 0 and 1 converted into a numpy array
desiredOutput = np.array(output)

# Boundary Line using omega
x = np.arange(-1, + 1.1, 0.5)
y1 = -(w0 + w1 * x)/w2

# PTA(Perceptrcon Training Algorithm)---------------------------------------------------------------------------------

# Vector containing the different values of eta
eta =[1, 0.1, 10]
# PICK THE NEW RANDOM WEIGHTS
print("NEW WEIGHTS")
# Pick w0_1 uniformly at random
w0_1 = np.random.uniform(-1, 1)
# Pick w1_1 uniformly at random
w1_1 = np.random.uniform(-1, 1)
# Pick w2_1 uniformly at random
w2_1 = np.random.uniform(-1, 1)

# Create omega2,the new vector of weights
omega_2 = np.asarray([w0_1, w1_1, w2_1])
print("NEW VECTOR OF WEIGHTS: " + str(omega_2))

# Boundary Line with new weights(omega')
y2 = -(w0_1 + w1_1 * x)/w2_1

tmp3 = []
tmp4 = []
actualOutput = []

# Dot product
for i in range(0, row):
    dot = np.dot(s[i, :], omega_2)
    if dot >= 0:
        actualOutput.append(1)
    else:
        actualOutput.append(0)

# Actual output containing 0 or 1 based on the new weights classification
actualOutput = np.array(actualOutput)


misclass=[]
# Calculate the number of misclassifications comparing the vectors actualOutput and desiredOutput
mis = np.count_nonzero(actualOutput != desiredOutput)

# Create a vector that will contain all the misclassifications

print("Total number of misclassifications using omega' " + str(mis))


print ("Initial Omega : " + str(omega))
print ("Omega' : " + str(omega_2))
omegaFinal = []
epochNum = []
num = []
y=[]

# Run the PTA for all the different values of eta
for i in range (0,len(eta)):
    print("i"+ str(i))
    results = Pta(omega,omega_2,mis,s,row)
    results.pta_alg(desiredOutput,actualOutput,eta[i])
    epochNum.append(results.get_epoch())
    misclass.append(results.get_misclass())
    num.append(results.get_num())
    omegaTmp = results.get_omega_final()
    print ("Final Omega : " + "with eta = " + str(eta[i]) + " " + str(omegaTmp))
    omegaFinal.append(omegaTmp)

    print ("eta leng" + str(len(eta)) + " e " + str(eta[i]))

    # Final Boundary Line
    y.append(-(omegaFinal[i][0] + omegaFinal[i][1] * x)/omegaFinal[i][2])

    # PLOT S0,S1 AND BOUNDARY --------------------------------------------------------------------------------

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # LABELS
    plt.title("PLOT BOUNDARIES WHEN ETA = " + str(eta[i]))
    plt.xlabel("X-AXIS(X1)")
    plt.ylabel("Y-AXIS(X2)")

    # Set limits of x and y axes
    axes = plt.gca()
    a=axes.set_xlim([-1, 1])
    b=axes.set_ylim([-1, 1])

    create_grid(ax,-1, 1.1, 0.2, 0.1)

    # Plot the boundary lines
    ax.plot(x, y1, linewidth='2', color='black', label='Original Boundary')
    ax.plot(x, y2, linewidth='2', color='red', label='Random Weights Boundary')
    ax.plot(x, y[i], linewidth='2', color='yellow', label='PTA Boundary')

    # Plot of S0 and S1, the if conditions are used in case the distribution is.....
    ax.scatter(s1[0:row, 0], s1[0:row, 1], color='red', s=70, linewidth='1', edgecolors='black', label='S1')
    ax.scatter(s0[0:row, 0], s0[0:row, 1], color='green', marker='v', s=70, linewidth='1', edgecolors='black', label='S0')
    ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., scatterpoints=1)

    # PLOT EPOCH NUMBER AND MISCLASSIFICATIONS--------------------------------------------------------------

    # Create a new figure
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)

    # LABELS
    plt.title("PLOT EPOCH NUMBER VS MISCLASSIFICATIONS WHEN ETA = " + str(eta[i]))
    plt.xlabel("EPOCH NUMBER")
    plt.ylabel("MISCLASSIFICATIONS")

    axes = plt.gca()
    a=axes.set_xlim([min(epochNum[i]), max(epochNum[i])])
    b=axes.set_ylim([min(misclass[i]), max(misclass[i])])

    plt.xticks(np.arange(min(epochNum[i]), max(epochNum[i]) + 1, 2.0))

    # Plot the epoch number vs the number of misclassifications
    ax2.plot(epochNum[i][:], misclass[i][:], linewidth='2', color='black', label='Boundary')
    plt.show()
