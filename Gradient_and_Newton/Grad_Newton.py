from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys

# Gradient calculation
def gradient(w):
    x=w[0]
    y=w[1]
    grad=np.zeros(2)

    # Formula calculated by hand
    comp1=1/(1-x-y)-1/x
    comp2=1/(1-x-y)-1/y
    grad[0]=comp1
    grad[1]=comp2

    return grad

# Calculate the Hessian Matrix
def hessian(w):
    x = w[0]
    y = w[1]

    # Formula calculated by hand
    fxx=1/pow((1-x-y),2) +1/pow(x,2)
    fxy = 1 / pow((1 - x - y), 2)
    fyx = 1 / pow((1 - x - y), 2)
    fyy = 1 / pow((1 - x - y), 2) + 1 / pow(y, 2)

    return np.array([[fxx,fxy],[fyx,fyy]])


# Apply gradient descent method
def gradient_descent(diff,i,old_w,weights,iter,energy_vec):
    # Use a threshold in order to allow the weights to converge
    while (diff[0] > threshold or diff[1] > threshold):
        i = i + 1
        # Gradient descent expression
        w = old_w - np.dot(eta, gradient(old_w))
        diff = abs(w - old_w)
        old_w = w

        # Append the iteration number in a vector
        iter = np.append(iter, i)
        # Append the weights in a vector
        weights.append(old_w)

        # Append the energy functions in a vector
        energy_vec = np.append(energy_vec, energy_function(old_w))

        # Check if the weights values is in the domain of the function
        if (old_w[0] + w[1] >= 1 or old_w[0] <= 0 or old_w[1] <= 0):
            sys.exit(" Out of domain, change your initial configuration")
    weights_vec = np.array(weights)

    return energy_vec,iter,weights_vec

# Apply newton's method
def newton_method(diff,i,old_w,weights,iter2,energy_vec2):
    # Use a threshold in order to allow the weights to converge
    while (diff[0] > threshold or diff[1] > threshold):
        i = i + 1
        weights.append(old_w)
        # Calculate the inverse of the hessian
        h_inverse=np.linalg.inv(hessian(old_w))
        # Newton's method expression
        w = old_w - np.dot(np.dot(eta, h_inverse), gradient(old_w))
        diff = abs(w - old_w)
        old_w = w

        # Append the iteration number in a vector
        iter2 = np.append(iter2, i)

        # Append the weights in a vector
        weights.append(old_w)

        # Append the energy functions in a vector
        energy_vec2 = np.append(energy_vec2, energy_function(old_w))

        # Check if the weights values is in the domain of the function
        if (old_w[0] + w[1] >= 1 or old_w[0] <= 0 or old_w[1] <= 0):
            sys.exit(" Out of domain, change your initial configuration")
    weights_vec2 = np.array(weights)

    return energy_vec2,iter2,weights_vec2


# Calculate the energy function given the weight
def energy_function(w):
    x = w[0]
    y = w[1]
    return -np.log(1-x-y) - np.log(x) - np.log(y)

# Initial weight
values=[1/2,1/5]
old_w=np.array(values)
eta=0.01
grad=[]
threshold=0.001

# Caclulate the gradient
grad=gradient(old_w)

w=np.zeros(2)
diff=old_w
weights=[]
weights2=[]
i=0
i2=0
a=[]
b=[]
iter=np.append(a,0)
iter2=np.append(a,0)

# Energy vectors that will contain the energy function for the gradient descent and newton method
energy_vec=np.append(b,energy_function(old_w))
energy_vec2=np.append(b,energy_function(old_w))

# Apply the gradient descent
energy_vec,iter,weights_vec =gradient_descent(diff,i,old_w,weights,iter,energy_vec)
print ("Final weights found with gradient descent : " + str(weights_vec[-1]))

# Initial weight
values=[1/2,1/5]
old_w2=np.array(values)
diff_2=old_w2


# Apply the newton method
energy_vec2,iter2,weights_vec2 =newton_method(diff_2,i2,old_w2,weights2,iter2,energy_vec2)
print ("Final weights found with newton's method : "+str(weights_vec2[-1]))

# PLOT-------------------------------------------------------------------------------------------

# Plot the figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


# LABELS
plt.title("WEIGTHS GRADIENT DESCENT")
plt.xlabel("W0(x)")
plt.ylabel("W0(y)")
axes = plt.gca()


ax.scatter(weights_vec[:,0], weights_vec[:,1], color='red', s=80, linewidth='1', edgecolors='black')

plt.show()

fig2= plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
plt.title("ENERGY FUNCTION")
plt.xlabel("ITERATIONS")
plt.ylabel("VALUES")
axes = plt.gca()
ax2.scatter(iter[:], energy_vec[:], color='blue', s=80, linewidth='1', edgecolors='black')
plt.show()


fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)

# LABELS
plt.title("WEIGTHS NEWTON'S METHOD")
plt.xlabel("W0(x)")
plt.ylabel("W0(y)")
axes = plt.gca()


ax3.scatter(weights_vec2[:,0], weights_vec2[:,1], color='red', s=80, linewidth='1', edgecolors='black')

plt.show()

fig4= plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
plt.title("ENERGY FUNCTION")
plt.xlabel("ITERATIONS")
plt.ylabel("VALUES")
axes = plt.gca()
ax4.scatter(iter2[:], energy_vec2[:], color='blue', s=80, linewidth='1', edgecolors='black')

plt.show()
