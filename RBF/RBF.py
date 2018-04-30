from __future__ import division
from numpy import linalg
import numpy as np
import random
import matplotlib.pyplot as plt
import pylab as pl

# Number of centers of the K-mean algorithm
n_centers=4
# Number of center for each class
n_center_one_class=2

# Calculate the labels given a set of input
def calculate_labels(x, d):
    for i in range(0,len(x)):
        # Apply the formula to calculate it
        if(x[i, 1]< (1/5)*np.sin(10*x[i,0])+0.3) or ((x[i, 1]-0.8)**2 + (x[i, 0]-0.5)**2 < 0.15**2):
            d[i] = 1
        else:
            d[i] = -1
    return d

# Returns an array containing the index to the nearest centroid for each point
def closest_centroid(points, centroids):
    # Calculate the distance
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


# Update the centroids
def update_centroids(points, closest, centroids):
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

# Change label of the class from -1 to 0
def change_label(all_points):
    new_points=[]
    for point in all_points:
        if point[-1]==-1 :
            point[-1]=0
        new_points.append(point)
    all_points=np.asarray(new_points)
    return all_points

# Gaussian RBF function with sigma=0.1 returns a vecotr of output given one input and a set of centroids
def RBF_function(point,centroids):
    sigma=0.1
    output=[]
    for center in centroids:
        output.append(np.exp(-linalg.norm(point - center) ** 2/ (2 * (sigma ** 2)) ))
    output=np.asarray(output)
    return output

# Gaussian RBF function with sigma=0.1 returns a single output given one input and one center
def RBF2_function(point,centroid):
    sigma=0.1
    output=np.exp(-linalg.norm(point - centroid) ** 2/ (2 * (sigma ** 2)) )
    return output


# Calculate the discriminant function g(x)
def discriminant_func(x,weights,centroids):
    g2=[]
    weights2=weights[1:]
    for point in x:
        g = 0
        for j in range(0,len(centroids)):
            # Discriminant function formula without bias
            g +=  weights2[j]* RBF2_function(point,centroids[j])
        # Add the bias
        g=g+weights[0]
        g2.append(g)
    g3=np.asarray(g2)
    return g3

# Step activation function
def activation_function(activate):
    if activate >=0:
        activate=1
    else:
        activate=0
    return activate

# PTA algorithm to find the weights
def PTA(all_points,centroids):
    epoch_vec=[]
    errors_vec=[]
    # Initialize at random m weights
    w = np.random.uniform(-1, 1, n_centers)
    # Add the bias
    weights = np.insert(w, 0, 1, axis=0)
    epoch_number = 0
    eta=1
    errors=1
    # Until the alogirthm does not converge
    while errors !=0 :
        epoch_number = epoch_number + 1
        epoch_vec.append(epoch_number)
        errors=0
        for point in all_points:
            # Calculate the PTA input
            pta_input = RBF_function(point[:-1], centroids)
            pta_input2 = np.insert(pta_input, 0, 1, axis=0)
            # Calculate the predicted label for the input
            dot=np.dot(pta_input2,weights)
            actual_y=activation_function(dot)
            desired=point[-1]
            # Update the weights
            weights=weights + eta*(desired-actual_y)*pta_input2
            # Calculate number of errors
            if(desired != actual_y):
                errors=errors+ 1
            else:
                errors=errors
        print ("epoch " + str(epoch_number))
        print ("errors " + str(errors))
        errors_vec.append(errors)
    final_errors=np.asarray(errors_vec)
    final_epochs=np.asarray(epoch_vec)
    return weights,final_epochs,final_errors

# Random number generator
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

# KMEANS-----------------------------------------------------------------------------------------------------------------------------------

#Pick 10 random centers from c1
centers_c1=random.sample(c1[:,:-1],n_center_one_class)
#Pick 10 random centers from c2
centers_c2=random.sample(c2[:,:-1],n_center_one_class)
centers_c1=np.asarray(centers_c1)
centers_c2=np.asarray(centers_c2)
no_errors=False
no_errors2=False

fig = plt.figure()

# PLOT before update --------------------------------------------------------------------------------

# LABELS
plt.title("BEFORE  UPDATE")
plt.xlabel("X1-AXIS(X1)")
plt.ylabel("X2-AXIS(X2)")

plt.scatter(centers_c1[:, 0], centers_c1[:, 1], color='red', s=150, linewidth='1', edgecolors='black',label="centroid C1")
plt.scatter(centers_c2[:, 0], centers_c2[:, 1], color='green', s=150, linewidth='1', edgecolors='black',label="centroid C2")
plt.scatter(c1[:, 0], c1[:, 1], color='red', s=20, linewidth='1', edgecolors='black', label='C1')
plt.scatter(c2[:, 0], c2[:, 1], color='green', marker='v', s=20, linewidth='1', edgecolors='black', label='C2')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., scatterpoints=1)

# K means centroids update for class c1
while no_errors != True:
    closest_c1=closest_centroid(c1[:,:-1],centers_c1)
    updated_centers_c1 = update_centroids(c1[:, :-1], closest_c1, centers_c1)
    if np.array_equiv(centers_c1, updated_centers_c1):
        no_errors= True
    centers_c1=updated_centers_c1
updated_centers_c1=centers_c1

# K means centroids update for class c2
while no_errors2 != True:
    closest_c2=closest_centroid(c2[:,:-1],centers_c2)
    updated_centers_c2 = update_centroids(c2[:, :-1], closest_c2, centers_c2)
    if np.array_equiv(centers_c2, updated_centers_c2):
        no_errors2= True
    centers_c2=updated_centers_c2
updated_centers_c2=centers_c2


# Put the updataed centroids together
total_centroids=np.vstack((updated_centers_c1, updated_centers_c2))

# Apply PTA ------------------------------------------------------------------------------------------------------------

# Change the label of -1 into 0
all_points=change_label(all_points)
# Calculate the new weights
weights,epoch_vec,errors_vec=PTA(all_points, total_centroids)


# PLOT epoch number and errors -----------------------------------------------------------------------------------------
fig = plt.figure()

# LABELS
plt.title("EPOCH N VS ERRORS")
plt.xlabel("EPOCH")
plt.ylabel("ERRORS")

plt.plot(epoch_vec, errors_vec, linewidth='1')



# PLOT updated centroids and decision boundary -------------------------------------------------------------------------
fig = plt.figure()
# LABELS
plt.title("AFTER UPDATE")
plt.xlabel("X1-AXIS(X1)")
plt.ylabel("X2-AXIS(X2)")

plt.scatter(updated_centers_c1[:, 0], updated_centers_c1[:, 1], color='red', s=150, linewidth='1', edgecolors='black',label="centroid C1")
plt.scatter(updated_centers_c2[:, 0], updated_centers_c2[:, 1], color='green', s=150, linewidth='1', edgecolors='black',label="centroid C2")
plt.scatter(c1[:, 0], c1[:, 1], color='red', s=20, linewidth='1', edgecolors='black', label='C1')
plt.scatter(c2[:, 0], c2[:, 1], color='green', marker='v', s=20, linewidth='1', edgecolors='black', label='C2')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., scatterpoints=1)
X1, X2 = np.meshgrid(np.linspace(0,1,300), np.linspace(0,1,300))
X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
g = discriminant_func(X,weights,total_centroids).reshape(300,300)
plt.contour(X1, X2, g, [0.001], colors='k', linewidths=1, origin='lower')

plt.show()