#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:57:26 2022

@author: dustinherrmann
"""

# import these basic modules - with these you can use most MATLAB functions in some form in python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import eigs

# experiment with matrices 
np.random.rand(3,2)
test2=np.eye(3)


B = np.array([[2,3,1],[5,6,6],[1,5,6]])

# select column as row vector 
B[:,0]

# select column as column vector 
B[:,[0]]

# conditional indexing 
np.argwhere(B[:,0]==5)


# find trues in conditions
x = np.arange(6).reshape(2,3)
these = np.nonzero(x>1)

x[these]

# convert matrix to row vector 
X = np.array([[1,2,3],[4,5,6],[7,8,9]])
m=X.flatten()
n=X.ravel()

# flip row/column VECTORS
b = np.array([1, 2, 3])
b = b[:,np.newaxis]

# transpose row/column of a 'matrix' = pandas DataFrame
Y = pd.DataFrame(X)
Y2= Y.transpose()

# transpose np.array
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
A.T

# indexing in a dataframe
Y2.loc[1,2]
Y2.loc[1,2]=66

#  reshaping matrix
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
total_elements = np.prod(A.shape)
B = A.reshape(1, total_elements) 
B

# concatanation 
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8, 9],[10,11,12]])

C = np.concatenate((A, B), axis=0) # axis indicates along which dimension 
C

# stacking (could also be done using np.concatenate)
a = np.array([1,2,3])
b = np.array([4,5,6])

np.column_stack([a,b])

np.row_stack([a,b])


# test 
y = np.array([1,2,3])
y+=y # in place assignment...
y+=2
y

# logical values 
e = 0
r = 1

e and r
e or r

# IMPORTANT: get size along a certain dimension 
a = np.array([1,2,3])
b = np.array([4,5,6])

c=np.column_stack([a,b])
c.shape[0]
c.shape[1]

# some indexing examples 

b[-1] # get last element of this vector

test=np.random.rand(9,9)
sliced = test[:5,:3] # index first 5! rows and first 3! columns, start indexing at zero but if you want 5 still have to ask for :5 
sliced.shape

sliced2 = test[-5:,-2:] # index last x rows/cols, when the ':' is after the integer, automatically assume to 'end'

# index specific rows and columns - here take rows 2, 4, 5 and columns 1, 3

test[np.ix_([1, 3, 4], [0, 2])]

# take every other row 
slice1 = test[0:9:2,:]
slice2 = test[::2,:] # basically short form for start:end:steps of 2 
slice2 = test[::-1,:] # reverse row order, similar to flipud in matlab


# np.r_

test[np.r_[:len(test),0]] # this is the elegant form 
test[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0])] # this is the elegant form 
#np.r_ is equivalent to the np.concatenate! 
C = np.concatenate((test[0:len(test):1,:], [test[0,:]]), axis=0) # this is the non-elegant but intuitive form 

 
np.r_[1:6,4:8] # row concatenation [,] in matlab 
np.c_[1,5] # row concatenation [;] in matlab 


# other
np.nonzero(a > 0.5)
test[(np.nonzero(test>0.5)[0],np.nonzero(test>0.5)[1])]


# IMPORTANT: numpy assigns by reference. if you set y to x and change y USING A FUNCTION BUT NOT BY OVERWRITING IT, x is ALSO changed!!!
x = np.array(range(0,8,2));
y = x;

x
y

y += 1;
x
y

#solution 
y = x.copy()
y = x[1, :].copy()


# flatten matrix into one dimensional array
y = x.flatten('F') # the F maintains the ordering as known from Matlab

tt=np.random.rand(9,9)
t1 = tt.flatten() # rows first
t2 = tt.flatten('F') #columns first 

# increasing vector!!!
a=np.arange(1., 11.)
a=np.r_[1.:11.]
a=np.c_[1.:11.]
np.arange(1.,11.)[:, np.newaxis] # an alternative to get a column vector

# zeros and ones 
np.zeros((3, 4))
np.zeros((3, 4, 5))
np.ones((3, 4))
np.eye(3)
np.diag(a)
np.diag(a, 0)

# spacing vectors 
np.linspace(1,3,4) # with values between 1 and 3 generate 4 values 

# meshgrid 
np.mgrid[0:9,0:6]
np.mgrid[0:9,0:6][0]
np.mgrid[0:9,0:6][1]

np.meshgrid(np.r_[0:9],np.r_[0:6])

np.ix_(np.r_[0:9.],np.r_[0:6.])

# repmat equivalent = tiling 
a = np.random.rand(2,3)
np.tile(a, (2, 3))


## concatenation!!
# columns 
a = np.c_[0:3]
b = np.c_[6:9]

c = np.c_[a,b] # option 1 
c = np.concatenate((a,b),axis=1) # option 2 
c = np.hstack((a,b)) # option 3

# rows 
a = np.r_[0:3]
b = np.r_[6:9]

c = np.r_[a,b] # option 1 
c = np.concatenate((a,b),axis=0) # option 2 
c = np.vstack((a,b)) # option 3

# max and min operations 
this =np.random.rand(6,3)
this.max() # all dimensions, abolsute max
this.max(0) # row dimension, will have n_col entries 
this.max(1) # column dimension, will have n_row entries 


this =np.random.rand(6,3)
that =np.random.rand(6,3)

np.maximum(this,that) # compares matrix elementwise and returns pairwise max


# logical 
a = 0
b = 1
np.logical_and(a,b)
np.logical_or(a,b)

a&b

# sorting 
np.sort(b) 
b.sort(axis=1)
np.unique(a)

a.squeeze()

3&4


# RESHAPING CAVE! 
x = np.random.rand(4,3)
x
y=x.reshape(3,4,order='F').copy() # I THINK the most MATLAB equivalent version: flatten matrix along the columns and impose new shape 
y
y=x.reshape(3,4,order='C').copy() # this is default python: flatten along rows and impose new matrix! make sure to check which one you need
y


























