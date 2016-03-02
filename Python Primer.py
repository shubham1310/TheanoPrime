
# coding: utf-8

# # Python

# ## Lists and Tuples

# ### Indexing into list

# In[1]:

l = [1, 2, 3] # make a list

l[1] # index into it


# ### Appending to a list

# In[ ]:

l.append(4) # add to it 
l


# ### Deleting an element

# In[ ]:

del l[1] 
l


# ### Inserting an element

# In[ ]:

l.insert(1, 3) # insert into it
l


# ### Tuples

# In[ ]:

t = (1, 3, 3, 4) # make a tuple

l == t


# ### List to Tuple

# In[ ]:

t2 = tuple(l)
t2 == t


# ## Dictionaries

# In[ ]:

Dict = {}
Dict[1] = 2
Dict['one'] = 'two'
Dict['1'] = '2'
Dict


# ### Keys in Dictionary

# In[ ]:

print "Dictionary keys"
print Dict.keys()

print "\nValue at 1 :"
print Dict['1']

print "\nValue at one"
print Dict['one']


one = 1
print "\nValue at 1"
print Dict[one]

print "\nIterate over keys"
for key in Dict.keys():
    print key

print "\nDelete key : 1"
del Dict[1]
print Dict


# # Classes and Function

# ## Functions

# In[ ]:

def printer(x):
    print x

def adder(x,y):
    return x+y

def square(x):
    return x**2

a = 2
b = 3
print "Lets print a:"
printer(a)
print "\nLets print a + b"
printer(adder(a,b))
print "\n So you can pass the return of a function to another function just like everywhere. \n Lets take it another step further "
printer(square(adder(a,b)))


# ## Classes

# In[ ]:

class student(object):
    
    def __init__(self,name = None ,age = None):
        if name == None:
            self.name = "Amartya"
        else:
            self.name = name
        
        if age == None:   
            self.age = 20
        else:
            self.age = age
    
    def update_name(self,name):
        self.name = name
    
    def update_age(self,age):
        self.age = age
    
    def inc_age(self):
        self.age = self.age + 1
    
    def return_info(self):
        temp = [self.name, self.age]
        return temp


# In[ ]:

Amartya = student()
print"Amartya:"
print vars(Amartya)

Bhuvesh = student("Bhuvesh", 21)

print "\nBhuvesh:"
print vars(Bhuvesh)

print "\nIncrementing Bhuvesh's age"
Bhuvesh.inc_age()
print vars(Bhuvesh)

print "\nMake Amartya  a baby"
Amartya.update_age(1)
print vars(Amartya)

print "\nA list of attributes of Amartya(Just to show what lists are)"
print Amartya.return_info()


# # Exceptions

# In[ ]:

print "Adding 2 and 3"
printer(adder(2,3))

print "\nAdding 'Amartya' and 'Bhuvesh'"
printer(adder("amartya","bhuvesh"))

print "\nBut say we want to practical and only add numbers , not people."

def adder(x,y):
    try:
        if type(x) != 'int' or type(x) != 'float' or type(y) != 'int' or type(y) != 'float':
            raise ValueError()
        else:
            return x+y
    except ValueError:
        print "Error!! Error!! You cant add people\n"

print "\nAdding 'Amartya' and 'Bhuvesh'"
printer(adder("amartya","bhuvesh"))


# # Starting Numpy

# In[ ]:

import numpy as np #Please don't forget this


# ## Basic types of arrays and matrices

# ### Zero Array and Zero Matrix

# In[ ]:

zeroArray = np.zeros(5)
print "Zero Array"
print zeroArray
print "\nZero Matrix:"
zeroArray = np.zeros([5,10])
print zeroArray


# ### Ones array and Ones Matrix

# In[ ]:

oneArray = np.ones(5)
print "Ones Array"
print oneArray
print "\nOnes Matrix:"
oneArray = np.ones([5,10])
print oneArray


# ### Identity Matrix

# In[ ]:

I = np.identity(5)
print "Identity Matrix"
print I


# ### Basic vector stuff

# In[ ]:

A = [1, 2, 3]
B = np.asarray(A)
C = [4,5,6]
D = np.asarray(C)


# In[ ]:

print "Elementwise Multiplication"
print B*D
print "\nElementwise Addition"
print B+D
print "\n Dot Product"
print np.dot(B,D)


# In[ ]:

print "Lets square each element in the array"
print [x**2 for x in C]
print "\n Lets do some more complicated function"

def updateX(x):
    x = x + 2
    x = np.log(x)
    x = np.power(x,2)
    return x

print [updateX(x) for x in C]


# ### Useful stuffs that make your life easy when coding stuffs.

# In[ ]:

print "Createing an array of numbers from 1 to 9"
A = np.arange(1,10)
print A

print "\n Reshape an array to matrix"
B = np.reshape(A,[3,3])
print B

print "\n Transpose the matrix"
C = np.transpose(B)
print C

print "\n Make elements less than 5 0"
C[C<5] = 0
print C



# In[ ]:

print "Summing up elements"
print "\n Each column"
print np.sum(C,axis=0)
print "\n Each row"
print np.sum(C,axis=1)


# In[ ]:


print "Mean of elements"
print "\n Each column"
print np.mean(C,axis=0)
print "\n Each row"
print np.mean(C,axis=1)


# In[ ]:

print "Product of  elements"
print "\n Each column"
print np.prod(C,axis=0)
print "\n Each row"
print np.prod(C,axis=1)


# # Finally Theano!

# In[ ]:

import theano
import theano.tensor as T


# In[ ]:

# Create the scalars
x = T.scalar()
y = T.scalar()


# In[ ]:

print "Add two numbers"
temp1 = x + y
# So this is how you add two "Symbolic variables" 

addTh = theano.function([x,y],temp1)
theano.pp(addTh.maker.fgraph.outputs[0])


# In[ ]:

print addTh(1,2)


# In[ ]:

print "Comparing two numbers"

temp1 = T.le(x, y)
compTh = theano.function([x,y],temp1)

theano.pp(compTh.maker.fgraph.outputs[0])
print compTh(4,3)


# In[ ]:

print "If else operator in Theano"
xgy = T.ge(x,y)
res = 2*x*xgy + (1 - xgy)*3*x


ifelse = theano.function([x,y],res)
print ""
print theano.pp(compTh.maker.fgraph.outputs[0])
print ""
print ifelse(5,4)


# In[ ]:

#Create the symbolic graph
z = x + y
w = z * x
a = T.sqrt(w)
b = T.exp(a)
c = a ** b
d = T.log(c)

uselessFunc = theano.function([x,y],d)
theano.pp(uselessFunc.maker.fgraph.outputs[0])


# In[ ]:

print uselessFunc(1,4)


# ## Where's the vector stuff

# In[ ]:

x = T.vector('x')
y = T.vector('y')

A = np.asarray([1,2,3])
B = np.asarray([4,5,6])


# In[ ]:

xdoty = T.dot(x,y)
xaddy = T.sum(x+y) 
dotfn = theano.function([x,y], xdoty)
print "Lets do dot product in theano"
print A,B,dotfn(A,B)

print "\nFunctions with more than one outputs"
dotaddfn = theano.function([x,y], [xdoty,xaddy])

print dotaddfn(A,B)
print "\n All element wise operations are similar to numpy"


# ### The famous logistic function

# In[ ]:

x = T.matrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)

print theano.pp(logistic.maker.fgraph.outputs[0])
logistic([[0, 1], [-1, -2]])


# ## The update comes in

# In[ ]:

state = theano.shared(0)
inc = T.iscalar('inc')

#Update the state by incrementing it with inc
accumulator = theano.function([inc], state, updates=[(state, state+inc)])


# In[ ]:

for i in range(0,10):
    accumulator(i)
    # In order to get the value of the accumulated
    print state.get_value()
    
# We can also set the value of a shared variable
state.set_value(0)


# ## As you might have guessed ML is a lot about updating parameters to achieve lowest cost
# 
# ## But then we need to choose what to update it with

# ## Gear up for some magic
# 
# ## Gradient Magic

# In[ ]:

a = T.scalar('a')
b = T.sqr(a)
c = T.grad(b,a)

gradfn = theano.function([a],c)
print theano.pp(gradfn.maker.fgraph.outputs[0])

print gradfn(4)


# In[ ]:

B = theano.shared(np.asarray([1.,2.]))
R = T.sqr(B).sum()
A = T.grad(R, B)

Z = theano.function([], R, updates={B: B - .1*A})
for i in range(10):
    print('cost function = {}'.format(Z()))
    print('parameters    = {}'.format(B.get_value()))
# Try to change range to 100 to see what happens

