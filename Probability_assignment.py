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

# In[3]:
import tkinter as tk
import turtle
import numpy as np
import random
import matplotlib.pyplot as plt
import pylab


np.random.seed(1000)


def sim(steps, list_steps, start, probability):
    arr = np.random.choice(a=list_steps, p=probability,
                           replace=True, size=steps)
    return start+np.sum(arr)


def play(generation, steps, start, list_steps=[-1, 0, 1], probability=None):
    output = []
    for i in range(generation):
        output.append(sim(steps, list_steps, start, probability))
    return output


output1 = play(1000, 1000, 0, probability=[0.7, 0.1, 0.2])
output2 = play(1000, 1000, 0, probability=[0.2, 0.1, 0.7])
output3 = play(1000, 1000, 0, probability=[0.33, 0.34, 0.33])

def setupscreen(scren='grid', scale=3):
    
    root = tk.Tk()
    START_WIDTH = 2000
    START_HEIGHT = 2000

    frame = tk.Frame(root, width=START_WIDTH, height=START_HEIGHT)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    xscrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
    xscrollbar.grid(row=1, column=0, sticky=tk.E+tk.W)

    yscrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    yscrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)

    canvas = tk.Canvas(frame, width=START_WIDTH, height=START_HEIGHT,
                        scrollregion=(0, 0, START_WIDTH, START_HEIGHT),
                        xscrollcommand=xscrollbar.set,
                        yscrollcommand=yscrollbar.set)

    canvas.grid(row=0, column=0, sticky=tk.N +
                tk.S+tk.E+tk.W)

    xscrollbar.config(command=canvas.xview)
    yscrollbar.config(command=canvas.yview)

    frame.pack()


    turt = turtle.RawTurtle(canvas)
    turt2 = turtle.RawTurtle(canvas)


    h = turtle.RawTurtle(canvas)
    w = turtle.RawTurtle(canvas)

    line_space = 10
    h.penup()
    w.penup()
    h.goto(-1000, 1000)
    w.goto(-1000, 1000)
    h.speed(0)
    w.speed(0)
    h.pd()
    w.pd()
    w.right(90)
    total_lines = 200
    Line_distance = total_lines*line_space
    for i in range(total_lines):
        if i == total_lines//2:

            for j in range(int(total_lines)):
                    h.fd(line_space)
                    h.write(abs(j-((total_lines//2)-1)))
                    w.fd(line_space)
                    w.write((total_lines//2)-j)

        else:

                h.fd(Line_distance)
                w.fd(Line_distance)

        h.pu()
        w.pu()
        h.goto(-1000, h.ycor()-line_space)
        w.goto(w.xcor()+line_space, 1000)
        h.pd()
        w.pd()
        
    if scren == 'circle':
        
        circle = turtle.RawTurtle(canvas)
        circle.pu()
        circle.goto(0,-1000)
        circle.pd()
        circle.speed(9)
        circle.circle(1000) 
    return turt,turt2

def simulation(steps, list_steps1, start1, probability1 , list_steps2 = None, start2 = None,probability2 = None ):
    bob, bob2 = setupscreen()   
    bob2.penup() 
    bob2.hideturtle()
    bob.pensize(3)
    bob.goto(start1[0]*10, start1[1]*10) 
    bob.pendown() # set starting position
    bob.speed(1)

    if start2!=None:
        bob2.pensize(3)
        bob2.showturtle()
        bob2.pencolor('blue')
        bob2.goto(start2[0]*10, start2[1]*10)  # set starting position
        bob2.speed(1)
        bob2.pendown()

    split1 = {} 
    split2 = {}
    total1 = 0
    for i in range(len(list_steps1)):
        split1[list_steps1[i]] = probability1[i]*100+total1
        total1 = total1 + probability1[i]*100
    
    if start2!=None:
        total2 = 0
        for i in range(len(list_steps2)):
            split2[list_steps2[i]] = probability2[i]*100+total2
            total2 = total2 + probability2[i]*100
       
  
    for x in range(steps):
            

            randnumber = random.randint(0, 1000)
            randnumber = randnumber % 100

            randnumber2 = random.randint(0, 1000)
            randnumber2 = randnumber2 % 100
            for i in split1:
                if split1[i] >= randnumber:
                    bob.fd(i*10)
                    break

            if start2!=None:
                for j in split2:               
                    if split2[j] >= randnumber2:
                        bob2.fd(j*10)
                        break
                    if bob.pos() == bob2.pos():
                        break
            
                if bob.pos() == bob2.pos():
                    print('Time for collision is '+ str(x)+" Time Units")
                    break

    print('Final cordinates are ('+ str(bob.xcor()//10)+","+str(bob.ycor()//10)+")")
        

    tk.mainloop()
    
#simulation(500,  [-1, 0, 1],(0, 0),[0.5, 0, 0.5]) # run for task 1
#simulation(500,  [-1, 0, 1],(0, 0),[0.5, 0, 0.5], [-1, 0, 1],(5,0), [0.5, 0, 0.5]) # run for task 2
plt.hist(output3, bins=50, alpha=0.5, label="Equal probabilities")
plt.legend(loc="best")
# plt.show()

print("For a distance of 1000 steps: ")
print("Probability= Equal, expected distance:", np.mean(output3))


#
#
# ***We can run this program using the function call with any given, number of steps, starting position and different probabilities.***
#

# In[4]:


print("Probability= [0.7,0.1,0.2],starting position=0, expected distance:", np.mean(
    output1))
print("Probability= [0.1,0.2,0.7],starting position=0, expected distance:", np.mean(
    output2))
plt.hist(output1, bins=50, alpha=0.5, label="[0.7,0.1,0.2]")
plt.hist(output2, bins=50, alpha=0.5, label="[0.1,0.2,0.7]")
plt.legend(loc="best")
# plt.show()


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


def sim(start_a, start_b,probability_a,probability_b):
    list_steps = [1,0,-1]
    pos_a = start_a
    pos_b = start_b
    time_count = 0
    while pos_a !=pos_b:
        time_count += 1
        pos_a += np.random.choice(a=list_steps,p=probability_a,replace=True)
        pos_b += np.random.choice(a=list_steps,p=probability_b,replace=True)
    return time_count

def play(generation, start,probability):
    output = []
    for i in range(generation):
        output.append(sim(start[0], start[1],probability[0],probability[1]))
    return output

output1 = play(100,[0,50],[[0.7,0.1,0.2],[0.2,0.4,0.4]])
output2 = play(100,[500,-200],[[0.7,0.1,0.2],[0.2,0.4,0.4]])

plt.hist(output3, bins=50,alpha=0.5,label="person_a:0, person_b:50")
plt.hist(output3, bins=50,alpha=0.5,label="person_a:500, person_b:-200")
plt.legend(loc="best")
plt.show()

print("For a distance of 1000 instances: ")
print("person_a:0, person_b:50", np.mean(output1))
print("person_a:500, person_b:-200", np.mean(output2))


# ---
#
# ### Task 3:
#
# Create a two-dimensional random walk model using simulation-based approach. The two-dimensional
# region in R 2 is a circular region of radius 100 units (1 unit can be 1 cm or 1 meter, or 1 km). We
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
# At the end of this task the trajectory of the node starting from the origin with the re-entry model will
# be demonstrated and assessed.

np.random.seed(1000)

def sim(steps,list_steps,start,probability):
    pos=start
    angles=(np.pi/180)*np.arange(0,361)
    magnitude=np.random.choice(a=list_steps,p=probability,replace=True,size=steps)
    
    ''' 
        #Issue 01
        Not sure what does it mean by discrete random variable in the range [0-2pi].
        0-2pi is a continuous range not an integer one which can be taken as discrete.
    '''
    angle=np.random.choice(a=angles,replace=True,size=steps)
    
    #Traverse the magnitudes and angles and add the changes to x and y
    for i in range(len(magnitude)):
        delta_y=magnitude[i]*np.sin(angle[i])
        delta_x=magnitude[i]*np.cos(angle[i])
        pos[0]=pos[0]+delta_x
        pos[1]=pos[1]+delta_y
        if np.sqrt((pos[0]**2)+(pos[1]**2))>(10000*np.pi):
            '''
            #Issue 02
            The player goes back to the origin when it hits the boundary line,
            This will be changed as per the litreture we find
            
            '''
            pos[0]=0
            pos[1]=0
    return pos[0],pos[1]
        

def play(generation,steps,start,list_steps=[0,0.5,1],probability=None):
    xVals=[]
    yVals=[]
    dist=[]
    for i in range(generation):
            x_cord,y_cord=sim(steps,list_steps,start,probability)
            xVals.append(x_cord)
            yVals.append(y_cord)
            dist.append(np.sqrt((x_cord**2)+(y_cord**2)))
    return xVals,yVals,dist


xVals,yVals,dist=play(1000,100,[0,0],probability=[0.1,0.1,0.8])
xVals0, yVals0, dist0=play(1000,200,[0,0],probability=[0.8,0.1,0.1])

pylab.plot(xVals, yVals, "r-" ,label = "Single player")

pylab.plot(xVals0, yVals0, "b^" ,label = "Single player")
pylab.title('Spots Visited on Walk ('+ "1000" + ' steps)')
pylab.xlabel('Steps East/West of Origin')
pylab.ylabel('Steps North/South of Origin')
pylab.legend(loc = 'best')


print("For a distance of 1000 steps: ")
print("Probability= biased towards 1, expected distance:",np.mean(dist))
print("Probability= biased towards 0, expected distance:",np.mean(dist0))





# %%


# #### Task 04
# 
# Repeat task 1 by assuming that the step size is a continuous uniform random variable between 0−1.Again, you may find it easier to model the PDF as a uniform random variable.
# 

# In[6]:


import numpy as np
import random 
import matplotlib.pyplot as plt

np.random.seed(1000)

def sim(steps,list_steps,start):
    arr=np.random.uniform(list_steps[0],list_steps[1],steps)
    return start+np.sum(arr)  

def play(generation,steps,start,list_steps=[-1,1]):
    output=[]
    for i in range(generation):
            output.append(sim(steps,list_steps,start))
    return output

output3=play(1000,1000,0)


plt.hist(output3,bins=50,alpha=0.5,label="Equal probabilities")
plt.legend(loc="best")
plt.show()

print("For a distance of 1000 steps: ")
print("Probability= Equal, expected distance:",np.mean(output3))



# ----
# #### Task 5
# 
# Repeat task 3 by assuming that the **step size and the orientation are continuous random variablesbetween 0−1 and 0−2π.**  Again,  you may find it easier to model the PDF as a uniform randomvariable.  Feel free to try other distributions

# In[11]:


import numpy as np
import random 
import matplotlib.pyplot as plt
import pylab

np.random.seed(1000)

def sim(steps,list_steps,start):
    pos=start
    magnitude=np.random.uniform(list_steps[0],list_steps[1],steps)
    angle=(np.pi/180)*np.random.uniform(0,361,steps)
    
    #Traverse the magnitudes and angles and add the changes to x and y
    for i in range(len(magnitude)):
        delta_y=magnitude[i]*np.sin(angle[i])
        delta_x=magnitude[i]*np.cos(angle[i])
        pos[0]=pos[0]+delta_x
        pos[1]=pos[1]+delta_y
        if np.sqrt((pos[0]**2)+(pos[1]**2))>(10000*np.pi):
            '''
            #Issue 02
            The player goes back to the origin when it hits the boundary line,
            This will be changed as per the litreture we find
            
            '''
            pos[0]=0
            pos[1]=0
    return pos[0],pos[1]
        

def play(generation,steps,start,list_steps=[0,1]):
    xVals=[]
    yVals=[]
    dist=[]
    for i in range(generation):
            x_cord,y_cord=sim(steps,list_steps,start)
            xVals.append(x_cord)
            yVals.append(y_cord)
            dist.append(np.sqrt((x_cord**2)+(y_cord**2)))
    return xVals,yVals,dist


xVals,yVals,dist=play(1000,10,[0,0])
xVals0, yVals0, dist0=play(1000,100,[0,0])
xVals1, yVals1, dist1=play(1000,1000,[0,0])

pylab.plot(xVals, yVals, "r-" ,label = "Single player")

pylab.plot(xVals0, yVals0, "b^" ,label = "Single player")

pylab.plot(xVals1, yVals1, "yo" ,label = "Single player")

pylab.title('Spots Visited on Walk ('+ "1000" + ' steps)')
pylab.xlabel('Steps East/West of Origin')
pylab.ylabel('Steps North/South of Origin')
pylab.legend(loc = 'best')


print("For a distance of 1000 steps: ")
print("10 steps, expected distance:",np.mean(dist))
print("100 steps, expected distance:",np.mean(dist0))
print("1000 steps, expected distance:",np.mean(dist1))



# In[ ]:


