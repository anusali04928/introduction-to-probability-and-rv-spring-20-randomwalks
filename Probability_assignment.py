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

import turtle
import numpy as np
import random
import matplotlib.pyplot as plt

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


def simulation(steps, start, list_steps, probability=None):
    window = turtle.Screen()  # Screen
    window.bgcolor("white")
    window.title("Walk of Life Simulation")
    bob = turtle.Turtle()  # Drawing tool
    bob.pensize(3)
    bob.pendown()
    bob.goto(start[0], start[1])  # set starting position
    bob.speed(1)

    split = {}
    total = 0
    setupscreen()
    for i in range(len(list_steps)):
        split[list_steps[i]] = probability[i]*100+total
        total = total+probability[i]*100

    for i in range(steps):
        randnumber = random.randint(0, 1000)
        randnumber = randnumber % 100
        for i in split:
            if split[i] >= randnumber:
                bob.fd(i*20)
                break
    turtle.Screen().exitonclick()


def setupscreen():

    h = turtle.Pen()
    w = turtle.Pen()
    Line_distance = 880
    line_space = 20
    h.penup()
    w.penup()
    h.goto(-400, 400)
    w.goto(-400, -400)
    h.speed(10)
    w.speed(10)
    h.pd()
    w.pd()
    w.left(90)
    total_lines = 40
    for i in range(total_lines):
        if i == total_lines//2:

            for j in range(Line_distance//(total_lines//2)):
                h.fd(line_space)
                h.write(j-(total_lines//2-1))
                w.fd(line_space)
                w.write(j-(total_lines//2))

        else:

            h.fd(Line_distance)
            w.fd(Line_distance)

        h.pu()
        w.pu()
        h.goto(-440, h.ycor()-line_space)
        w.goto(w.xcor()+line_space, -440)
        h.pd()
        w.pd()


simulation(100, (0, 0), [-1, 0, 1], [0.5, 0, .5])
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


def sim(start_a,start_b,probability_a,probability_b):
    list_steps=[1,0,-1]
    pos_a=start_a
    pos_b=start_b
    time_count=0
    while pos_a!=pos_b:
        time_count+=1
        pos_a+=np.random.choice(a=list_steps,p=probability_a,replace=True)
        pos_b+=np.random.choice(a=list_steps,p=probability_b,replace=True)
    return time_count

def play(generation,start,probability):
    output=[]
    for i in range(generation):
            output.append(sim(start[0],start[1],probability[0],probability[1]))
    return output

output1=play(100,[0,50],[[0.7,0.1,0.2],[0.2,0.4,0.4]])
output2=play(100,[500,-200],[[0.7,0.1,0.2],[0.2,0.4,0.4]])

plt.hist(output3,bins=50,alpha=0.5,label="person_a:0, person_b:50")
plt.hist(output3,bins=50,alpha=0.5,label="person_a:500, person_b:-200")
plt.legend(loc="best")
plt.show()

print("For a distance of 1000 instances: ")
print("person_a:0, person_b:50",np.mean(output1))
print("person_a:500, person_b:-200",np.mean(output2))


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

# Yet to be done, I could not do it yesterday.
