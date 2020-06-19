#!/usr/bin/env python
# coding: utf-8

# # Probability & Statistics Project
# 
# ### Task 1:
# 
# Create a one-dimensional random walk mathematical and simulation-based model to predict the ex-
# pected distance from the starting point. You should test your models for range of scenarios e.g.
# **starting position**, **(un)equal probabilities** of moving left, right, and not moving at all.
# 

# In[2]:


import numpy as np
import random 
import matplotlib.pyplot as plt

np.random.seed(10)

def sim(steps,list_steps,start,probability):
    arr=np.random.choice(a=list_steps,p=probability,replace=True,size=steps)
    return start+np.sum(arr)  

def play(generation,steps,start,list_steps=[-1,0,1],probability=None):
    output=[]
    for i in range(generation):
            output.append(sim(steps,list_steps,start,probability))
    return output

def plot(outcomes,label_text,information):
    print(information,np.mean(outcomes))
    plt.hist(outcomes,bins=50,alpha=0.5,label=label_text)


output0=play(100000,100,0,probability=[0.33,0.34,0.33])
output1=play(100000,100,0,probability=[0.7,0.1,0.2])
output2=play(100000,100,0,probability=[0.2,0.1,0.7])

plot(output0,"Equal Probability","starting position=0,probability used= [0.33,0.34,0.33] , expected distance:")
plot(output1,"Unequal Probability","starting position=0,probability used= [0.7,0.1,0.2] , expected distance:")
plot(output2,"Unequal Probability","starting position=0,probability used= [0.1,0.2,0.7] , expected distance:")
plt.legend(loc="best")
plt.show()


# 
# 
# ***We can run this program using the function call with any given, number of steps, starting position and different probabilities.***
# 

# --- 
# 
# ### Task 2
# 
# Create a one-dimensional random walk mathematical and simulation-based model where **two people
# start x units away from each other** and are **traversing the grid with some probabilities**. Predict the
# expected time it takes to meet them. You should test your models for a range of scenarios. List any
# assumptions you may have made.
# 
# **Asumption: Each step takes one second**

# In[ ]:



import numpy as np
import random
import matplotlib.pyplot as plt
import pylab


def sim(start_a, start_b,probability_a,probability_b):
    list_steps = [1,0,-1]
    pos_a = start_a
    pos_b = start_b
    time_count = 0
    while pos_a !=pos_b:
        time_count += 1
        pos_a += np.random.choice(a=list_steps,p=probability_a,replace=True)
        if pos_a == pos_b:
            break
        pos_b += np.random.choice(a=list_steps,p=probability_b,replace=True)
    return time_count

def play(generation, start,probability):
    output = []
    for i in range(generation):
        output.append(sim(start[0], start[1],probability[0],probability[1]))
    return output

output3 = play(100000,[-10,10],[[0.7,0.1,0.2],[0.2,0.4,0.4]])


plt.hist(output3, bins=50,alpha=0.5,label="person_a:-10, person_b:10")
plt.legend(loc="best")
plt.show()

print("For a distance of 1000 instances: ")
print("person_a:-10, person_b:10", np.mean(output3))


# ---
# 
# ### Task 3:
# 
# Create a two-dimensional random walk model using simulation-based approach. The two-dimensional
# region in R^2 is a circular region of radius 100 units (1 unit can be 1 cm or 1 meter, or 1 km). We
# would refer our two dimensional circular space as a disc, d(0, R). A node or nodes within a circular
# region d(0, R) at time t, undertakes a random walk whose next position at the fixed discrete time t + 1
# unit time is modeled as follows:
# 
# - Step size is a discrete random variable. You can assume that step size is between {0, 0.5, 1}.
# - Orientation is a discrete random variable between [0 − 2π].
# 
# For example – a node at the center of a 100-unit radius circle will take a random step of size r between
# {0, 0.5, 1} and a direction with angle θ. To begin with, you may find it easier to assume that all step
# sizes and angles are equally likely.
# 
# Over the next course of time slots the node will traverse a random path based on the random walk
# model described above. Of course, at some point in time, it is quite likely that after many units of
# time, the node might try to leave the 100-unit circular region. Its re-entry model, to ensure that
# the node does not escape the test region, is left for the students to be worked-out. As an example,
# once the node hits the boundary you can ensure that its bounces off the circumference back into the
# region. Whatever model of node re-entry you choose, it should be based on logic, explanation and
# some literature review. You would be asked for its explanation and justification.
# **At the end of this task the trajectory of the node starting from the origin with the re-entry model will
# be demonstrated and assessed.**

# In[47]:


import numpy as np
import random 
import matplotlib.pyplot as plt
import pylab
import math

np.random.seed(10)

def boundary(pos,delta_x,delta_y,prev_pos):
    #Bringing the point back to boundary
    overstep=np.round(np.sqrt((pos[0]**2)+(pos[1]**2))-100)
    if delta_x==0 or delta_y==0:
        return pos[0],pos[1]
    overstep_x=overstep*(delta_x/(delta_x+delta_y))
    overstep_y=overstep*(delta_y/(delta_x+delta_y))
    #Calculating angle
    m1=((pos[1]-prev_pos[1])/pos[0]-prev_pos[0])
    m2=pos[1]/pos[0]
    angle=(np.pi)-math.atan(abs((m1-m2)/(1+(m1*m2))))
    magnitude=np.sqrt((overstep_x**2)+(overstep_y**2))
    #Reflect
    delta_y=magnitude*np.sin(angle)
    delta_x=magnitude*np.cos(angle)
    pos[0]=pos[0]-overstep_x+delta_x
    pos[1]=pos[1]-overstep_y+delta_y
    return pos[0],pos[1]
    
def sim(steps,list_steps,start,probability):
    plot_pts=[[],[]]
    pos=start
    angles=(np.pi/180)*np.arange(0,361)
    magnitude=np.random.choice(a=list_steps,p=probability,replace=True,size=steps)
    angle=np.random.choice(a=angles,replace=True,size=steps)
    
    #Traverse the magnitudes and angles and add the changes to x and y
    for i in range(steps):
        if np.sqrt((pos[0]**2)+(pos[1]**2))>100:
            pos[0],pos[1]=boundary(pos,delta_x,delta_y,prev_pos)
        delta_y=round(magnitude[i]*np.sin(angle[i]),2)
        delta_x=round(magnitude[i]*np.cos(angle[i]),2)
        prev_pos=pos
        pos[0]=round(pos[0]+delta_x,2)
        pos[1]=round(pos[1]+delta_y,2)
        #Reflective boundary
        if np.sqrt((pos[0]**2)+(pos[1]**2))>100:
            pos[0],pos[1]=boundary(pos,delta_x,delta_y,prev_pos)
        plot_pts[0].append(pos[0])
        plot_pts[1].append(pos[1])
    return plot_pts
        
def play(generation,steps,start,list_steps=[0,0.5,1],probability=None):
    plots=[]
    for i in range(generation):
            plot=sim(steps,list_steps,start,probability)
            plots.append(plot)
    return plots

x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 10000
plt.contour(X,Y,F,[0])
def draw_path(plot,style):
        for i in plot:
            pylab.plot(i[0],i[1] , style ,label = "Single player")

draw_path(play(1,1000,[0,0]),"y-")           
draw_path(play(1,10000,[0,0]),"r-")
draw_path(play(1,900000,[0,0]),"b-")



pylab.title('Random walk in 2D with boundary of 100 units')
pylab.xlabel('Steps East/West of Origin')
pylab.ylabel('Steps North/South of Origin')
pylab.legend(loc = 'best')


print("For a distance of 1000 steps: ")


# #### Task 04
# 
# Repeat task 1 by assuming that the step size is a continuous uniform random variable between 0−1.Again, you may find it easier to model the PDF as a uniform random variable.
# 

# In[4]:


import numpy as np
import random 
import matplotlib.pyplot as plt

np.random.seed(1000)

def sim(steps,range_steps,start):
    arr=np.random.uniform(range_steps[0],range_steps[1],steps)
    return start+np.sum(arr)  

def play(generation,steps,start,range_steps=[0,1]):
    output=[]
    for i in range(generation):
            output.append(sim(steps,range_steps,start))
    return output


def plot(outcomes,label_text,information):
    print(information,np.mean(outcomes))
    plt.hist(outcomes,bins=50,alpha=0.5,label=label_text)


output0=play(100000,100,0)
output1=play(100000,100,10)
output2=play(100000,100,100)
output4=play(100000,1000,0)
output5=play(100000,1000,10)
output6=play(100000,1000,100)



print("For a distance of 1000 steps: ")
plot(output0,"starting position=0","starting position=0, expected distance:")
plot(output1,"starting position=10","starting position=10, expected distance:")
plot(output2,"starting position=100","starting position=100, expected distance:")
plt.legend(loc="best")
plt.show()

print("For a distance of 10000 steps: ")
plot(output4,"starting position=0","starting position=0, expected distance:")
plot(output5,"starting position=10","starting position=10, expected distance:")
plot(output6,"starting position=100","starting position=100, expected distance:")

plt.legend(loc="best")
plt.show()


# ----
# #### Task 5
# 
# Repeat task 3 by assuming that the **step size and the orientation are continuous random variablesbetween 0−1 and 0−2π.**  Again,  you may find it easier to model the PDF as a uniform randomvariable.  Feel free to try other distributions

# In[4]:


import numpy as np
import random 
import matplotlib.pyplot as plt
import pylab
import math


np.random.seed(100)
def boundary(pos, prev_pos,delta_x,delta_y):
    #Bringing the point back to boundary
    overstep=np.sqrt((pos[0]**2)+(pos[1]**2))-100
    overstep_x=overstep*(delta_x/(delta_x+delta_y))
    overstep_y=overstep*(delta_y/(delta_x+delta_y))
    #Calculating angle
    m1=pos[1]-prev_pos[1]/pos[0]-prev_pos[0]
    m2=pos[1]/pos[0]
    angle=(np.pi)-math.atan(abs((m1-m2)/(1+(m1*m2))))
    magnitude=np.sqrt((overstep_x**2)+(overstep_y**2))
    #Reflect
    delta_y=magnitude*np.sin(angle)
    delta_x=magnitude*np.cos(angle)
    pos[0]=pos[0]-overstep_x+delta_x
    pos[1]=pos[1]-overstep_y+delta_y
    return pos[0],pos[1]

    
def sim(steps,list_steps,start):

    plot_pts=[[],[]]
    pos=start
    magnitude=np.random.uniform(list_steps[0],list_steps[1],steps)
    angle=(np.pi/180)*np.random.uniform(0,361,steps)
    
    #Traverse the magnitudes and angles and add the changes to x and y
    for i in range(len(magnitude)):
        prev_pos=pos
        if np.sqrt((pos[0]**2)+(pos[1]**2))>=100:
            pos[0],pos[1]=boundary(pos,prev_pos,delta_x,delta_y)
        delta_y=magnitude[i]*np.sin(angle[i])
        delta_x=magnitude[i]*np.cos(angle[i])
        pos[0]=pos[0]+delta_x
        pos[1]=pos[1]+delta_y
        #Reflective boundary
        if np.sqrt((pos[0]**2)+(pos[1]**2))>=100:
            pos[0],pos[1]=boundary(pos,prev_pos,delta_x,delta_y)
        plot_pts[0].append(pos[0])
        plot_pts[1].append(pos[1])
    return plot_pts
        

def play(generation,steps,start,list_steps=[0,0.5,1]):
    plots=[]
    for i in range(generation):
            plot=sim(steps,list_steps,start)
            plots.append(plot)
    return plots

x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 10000
plt.contour(X,Y,F,[0])
def draw_path(plot,style):
        for i in plot:
            pylab.plot(i[0],i[1] , style ,label = "Single player")
            
draw_path(play(1,1000,[0,0]),"y-")
draw_path(play(1,100000,[50,0]),"m-")



pylab.title('Random walk in 2D with boundary of 100 units')
pylab.xlabel('Steps East/West of Origin')
pylab.ylabel('Steps North/South of Origin')
pylab.legend(loc = 'best')


print("For a distance of 1000 steps: ")


# ---
# 
# ### Task 07
# 
# 
# Repeat task 3 by assuming that the step size is a discrete random variable and the orientation is acontinuous random variables between 0−2π.
# ---

# In[7]:


import numpy as np
import random 
import matplotlib.pyplot as plt
import pylab
import math

np.random.seed(100)

def boundary(pos, prev_pos,delta_x,delta_y):
    #Bringing the point back to boundary
    overstep=np.sqrt((pos[0]**2)+(pos[1]**2))-100
    overstep_x=overstep*(delta_x/(delta_x+delta_y))
    overstep_y=overstep*(delta_y/(delta_x+delta_y))
    #Calculating angle
    m1=pos[1]-prev_pos[1]/pos[0]-prev_pos[0]
    m2=pos[1]/pos[0]
    angle=(np.pi)-math.atan(abs((m1-m2)/(1+(m1*m2))))
    magnitude=np.sqrt((overstep_x**2)+(overstep_y**2))
    #Reflect
    delta_y=magnitude*np.sin(angle)
    delta_x=magnitude*np.cos(angle)
    prev_pos=pos
    pos[0]=pos[0]-overstep_x+delta_x
    pos[1]=pos[1]-overstep_y+delta_y
    return pos[0],pos[1],prev_pos

    
def sim(steps,list_steps,start,probability):
    plot_pts=[[],[]]
    pos=start
    magnitude=np.random.choice(a=list_steps,p=probability,replace=True,size=steps)
    angle=(np.pi/180)*np.random.uniform(0,361,steps)
    
    #Traverse the magnitudes and angles and add the changes to x and y
    for i in range(len(magnitude)):
        if np.sqrt((pos[0]**2)+(pos[1]**2))>=100:
            pos[0],pos[1],prev_pos=boundary(pos,prev_pos,delta_x,delta_y)
        delta_y=magnitude[i]*np.sin(angle[i])
        delta_x=magnitude[i]*np.cos(angle[i])
        prev_pos=pos
        pos[0]=round((pos[0]+delta_x),2)
        pos[1]=round((pos[1]+delta_y),2)
        #Reflective boundary
        if np.sqrt((pos[0]**2)+(pos[1]**2))>=100:
            pos[0],pos[1],prev_pos=boundary(pos,prev_pos,delta_x,delta_y)
        plot_pts[0].append(pos[0])
        plot_pts[1].append(pos[1])
    return plot_pts
        

def play(generation,steps,start,list_steps=[0,0.5,1],probability=None):
    plots=[]
    for i in range(generation):
            plot=sim(steps,list_steps,start,probability)
            plots.append(plot)
    return plots

x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 10000
plt.contour(X,Y,F,[0])
def draw_path(plot,style):
        for i in plot:
            pylab.plot(i[0],i[1] , style ,label = "Single player")
            


draw_path(play(1,100000,[0,0]),"m-")
draw_path(play(1,10000,[0,0]),"b-")

pylab.title('Random walk in 2D with boundary of 100 units')
pylab.xlabel('Steps East/West of Origin')
pylab.ylabel('Steps North/South of Origin')
pylab.legend(loc = 'best')


print("For a distance of 1000 steps: ")


# ### Task 08
# 
# Building  on  task  5,  each  team  will  capture  the  trajectory  of  two  nodes  whose  initial  locations  arechosen randomly and uniformly over a circular region.  Every team will be asked to explain how didthey model the initial position of the two nodes.  In this part you will have to determine, the averagenumber  of  steps  taken  by  the  two  nodes  so  that  they  are  within  1  unit  (distance  between  the  twonodes<1 unit distance).  You will need to work out the number of simulations required to calculatethe average number of steps needed.  There is some science behind it.  So make sure you are readyto defend the accuracy of your expected value of the number of steps taken by the two nodes so thatthey are within 1 unit of each other.

# In[21]:


import numpy as np
import random 
import matplotlib.pyplot as plt
import pylab
import math

np.random.seed(1000000)

def boundary(pos,delta_x,delta_y,prev_pos):
    #Bringing the point back to boundary
    overstep=np.sqrt((pos[0]**2)+(pos[1]**2))-100
    overstep_x=overstep*(delta_x/(delta_x+delta_y))
    overstep_y=overstep*(delta_y/(delta_x+delta_y))
    #Calculating angle
    m1=pos[1]-prev_pos[1]/pos[0]-prev_pos[0]
    m2=pos[1]/pos[0]
    angle=(np.pi)-math.atan(abs((m1-m2)/(1+(m1*m2))))
    magnitude=np.sqrt((overstep_x**2)+(overstep_y**2))
    #Reflect
    delta_y=magnitude*np.sin(angle)
    delta_x=magnitude*np.cos(angle)
    pos[0]=pos[0]-overstep_x+delta_x
    pos[1]=pos[1]-overstep_y+delta_y
    return pos[0],pos[1]

    
def move(list_steps,pos):
    magnitude=np.random.uniform(list_steps[0],list_steps[1],1)
    angle=(np.pi/180)*np.random.uniform(0,361,1)
    delta_y=magnitude[0]*np.sin(angle[0])
    delta_x=magnitude[0]*np.cos(angle[0])
    pos[0]=round((pos[0]+delta_x),2)
    pos[1]=round((pos[1]+delta_y),2)
    return pos,[delta_x,delta_y]

def sim(list_steps,start):
    iterations=0
    pos0=[start[0],start[1]]
    pos1=[start[2],start[3]]
    plot_pts0=[[],[]]
    plot_pts1=[[],[]]
    #Traverse the magnitudes and angles and add the changes to x and y
    while  np.sqrt(((pos0[0]-pos1[0])**2)+((pos0[1]-pos1[1])**2))>1:
        prev_pos0=pos0
        prev_pos1=pos1
        pos0,delta0=move(list_steps,pos0)
        if np.sqrt(((pos0[0]-pos1[0])**2)+((pos0[1]-pos1[1])**2))<=1:
            break
        pos1,delta1=move(list_steps,pos1)
        iterations+=1
        
        plot_pts0[0].append(pos0[0])
        plot_pts0[1].append(pos0[1])
        plot_pts1[0].append(pos1[0])
        plot_pts1[1].append(pos1[1])
    return iterations, plot_pts0, plot_pts1
        

def play(generation,list_steps=[0,1]):
    iterations=[]
    plot0=0
    plot1=0
    iteration=0
    for i in range(generation):
            start=np.round(np.random.uniform(-10,10,4),2)

            iteration,plot0,plot1=sim(list_steps,start)
            iterations.append(iteration)
    return iterations, plot0, plot1, start


iterations,plot0,plot1,start=play(1)
x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 10000
plt.contour(X,Y,F,[0])

def draw_path(plot,style):
            pylab.plot(plot[0],plot[1] , style ,label = "Single player")

            
draw_path(plot0,"r-")
draw_path(plot1,"b-")
print("Steps taken:", iterations)
print("Starting position node 1: ",[start[0],start[1]])
print("Starting posiyion node 2: ",[start[2],start[3]])

pylab.title('Random walk in 2D with boundary of 100 units')
pylab.xlabel('Steps East/West of Origin')
pylab.ylabel('Steps North/South of Origin')
pylab.legend(loc = 'best')

iterations0,plot2,plot3,start=play(100000)
plt.hist(iterations0,bins=50,alpha=0.5,label=label_text)
plot(iterations,"100000 simulations")
plt.legend(loc="best")
plt.show()


# In[ ]:




