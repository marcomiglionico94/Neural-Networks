from __future__ import division
from numpy import linalg
import cvxopt
import cvxopt.solvers
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

# Sigma value used in the kernel
sigma = 1


# Calculate the labels given a set of input
def calculate_labels(x, d):
    for i in range(0,len(x)):
        # Apply the formula to calculate it
        if(x[i, 1]< (1/5)*np.sin(10*x[i,0])+0.3) or ((x[i, 1]-0.8)**2 + (x[i, 0]-0.5)**2 < 0.15**2):
            d[i] = 1
        else:
            d[i] = -1
    return d

# Apply the Gaussian Kernel given a sigma
def gaussian_kernel(x, x1, sigma):
    return np.exp(-linalg.norm(x-x1)**2 / (2 * (sigma ** 2)))


# Calculate the discriminant function
def discriminant_func(x,alpha,sv_d,sv,b):
    y_predict = np.zeros(len(x))
    for i in range(len(x)):
        s = 0
        for alpha2, sv_d2, sv2 in zip(alpha, sv_d, sv):
            # Discriminant function formula without bias
            s += alpha2 * sv_d2 * gaussian_kernel(x[i],sv2,sigma)
        y_predict[i] = s
    return y_predict + b

# Bias calculation
def calculate_bias(alpha,sv_d,sv):
    b2 = 0
    for n in range(len(alpha)):
        b2 += sv_d[n]
        b2 -= np.sum(alpha * sv_d * k[ind[n], sv])
    b2 /= len(alpha)
    return b2

# Predict the labels of the points
def predict(x,alpha,sv_d,sv,b):
    return np.sign(discriminant_func(x,alpha,sv_d,sv,b))


# Random number generator[i]
np.random.seed(2)

# Create 100 points x uniformly at random on [0,1]
x = np.random.uniform(0, 1, 200).reshape(100, 2)


# Labels output
d = np.zeros(100)

# Calculate the labels for every input
d=calculate_labels(x, d)
d=d.reshape(100, 1)


# Put the points and their class together in one vector
all_points=np.append(x,d,axis=1)
ones = (all_points[:, -1] == 1)
zeros = (all_points[:, -1] == -1)
# Class c1
c1 = all_points[ones]
# Class c2
c2 = all_points[zeros]

# Divide the point and the lables
x1=c1[:,:-1]
x2=c2[:,:-1]
y1=c1[:,-1]
y2=c2[:,-1]

# Unify them in order to have just 2 vectors
x = np.vstack((x1, x2))
d = np.hstack((y1,y2))


# Calculate the gaussian kernel
k = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        # Apply the gaussian kernel
        k[i, j] = gaussian_kernel(x[i], x[j],sigma)

# CVXOPT SOLVER--------------------------------------------------------------------------------------------------------------------------
P = cvxopt.matrix(np.outer(d,d)*k)
q = cvxopt.matrix(np.ones(100)*-1)
A = cvxopt.matrix(d,(1,100))
b = cvxopt.matrix(0.0)
G = cvxopt.matrix(np.diag(np.ones(100)*-1))
h = cvxopt.matrix(np.zeros(100))

# Solve the quadratic function
solution=cvxopt.solvers.qp(P,q,G,h,A,b)

#-----------------------------------------------------------------------------------------------------------------------------

# The solution represent the alphas
alpha= np.ravel(solution['x'])


# Find the support vectors
sv= alpha > 1e-5
ind=np.arange (len(alpha))[sv]
alpha=alpha[sv]

# Support vectors points
sv_x=x[sv]

# Support vectors labels
sv_d= d[sv]

sv_d2=sv_d.reshape(14,1)
all_supports=np.append(sv_x,sv_d2,axis=1)
ones = (all_supports[:, -1] == 1)
zeros = (all_supports[:, -1] == -1)
supp_c1=all_supports[ones]
supp_c2=all_supports[zeros]

print (str(len(alpha)) + " support vectors out of 100 ")

# Calculate the optimal BIAS
b2= calculate_bias(alpha,sv_d,sv)

# Test it using the training labels
y_test= d

# Predict the labels
y_predict = predict(x,alpha,sv_d,sv_x,b2)

fig = plt.figure()

# LABELS
plt.title("SVM WITH GAUSSIAN KERNEL")
plt.xlabel("X1-AXIS(X1)")
plt.ylabel("X2-AXIS(X2)")

plt.scatter(c1[:, 0], c1[:, 1], color='red', s=50, linewidth='1', edgecolors='black', label='C1')
plt.scatter(c2[:, 0], c2[:, 1], color='orange', marker='v', s=50, linewidth='1', edgecolors='black', label='C2')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., scatterpoints=1)

plt.show

# Calculate the accuracy
correct = np.sum(y_predict == y_test)
print ("The number of correct prediction is " + str(correct) + " over " + str(len(y_predict)))
accuracy= (correct/len(y_predict)) * 100
print ("Accuracy : " + str(accuracy) + "%")


# PLOT-----------------------------------------------------------------------------------------------------------------------
# Create a new figure
fig = plt.figure()

# LABELS
plt.title("SVM WITH GAUSS IAN KERNEL")
plt.xlabel("X1-AXIS(X1)")
plt.ylabel("X2-AXIS(X2)")

plt.scatter(supp_c1[:, 0], supp_c1[:, 1], color='b', s=150, linewidth='1', edgecolors='black',label="support")
plt.scatter(supp_c2[:, 0], supp_c2[:, 1], color='b', marker='v', s=150, linewidth='1', edgecolors='black',label="support")
plt.scatter(c1[:, 0], c1[:, 1], color='red', s=50, linewidth='1', edgecolors='black', label='C1')
plt.scatter(c2[:, 0], c2[:, 1], color='orange', marker='v', s=50, linewidth='1', edgecolors='black', label='C2')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., scatterpoints=1)


X1, X2 = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = discriminant_func(X,alpha,sv_d,sv_x,b2).reshape(X1.shape)
pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

plt.show()
