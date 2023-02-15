import numpy as np
import networkx as nx
import scipy as sp
from ripser import ripser
import persim
import re
from matplotlib import pyplot as plt

#Basically get a list of strings with each graphs PH intervals(and some other garbage)

def Masterlist():
    #THIS LINE MUST BE CHANGED BASED ON THE FILE PATH IN YOUR COMPUTER
    with open("/home/mokuda/familynetworksresearch/FamilyNetworksResearch/Model/ORIGINAL_SOURCES_PH") as ofile:
        data = ofile.read()
        data = data.replace('\n',"")
        pattern = re.compile(r'kinsources(.*?)Boundary')
        graphs_text = pattern.findall(data)
        return graphs_text

masterlist = Masterlist()

#extract intervals from string and put intervals into a single array

def bottleneck_format(string):
    lines = re.compile(r"\[.{0,10}\]\sx\s\d*")
    #put all intervals into a list
    things = lines.findall(string)
    new_things = []

    #remove all syntax form interval except the actual birth and death times
    for thing in things:
        thing = thing.replace(".","")
        thing = thing.replace("[","")
        thing = thing.replace("]","")
        thing = thing.replace(",","")
        thing = thing.replace("x","")

        #get list including the birth time and death time
        split = thing.split()
        split[0] =  int(split[0])

        #cast as integers where needed
        if split[1] != "inf":
            split[1] = int(split[1])
        else:
            split[1] = np.inf

        split[2] = int(split[2])

        #reformat and put into array
        for i in range(split[2]):
            new_things.append(np.array([split[0],split[1]]))
            
    return np.array(new_things)



#create zero dimension array
def dgms0():
    k = 0
    dgms = []

    for i in masterlist:
        pattern = re.compile(r'DIMENSION 0(.*?)DIMENSION 1')
        masterlist0 = pattern.findall(i)
        dgms.append(bottleneck_format(masterlist0[0]))
        k += 1

    return dgms

dgms0 = dgms0()

#remove empty array from list of intervals
dgms0 = [i for i in dgms0 if len(i) > 0]

#combine all of the zero dimension arrays

A = dgms0[0]

for i in range(len(dgms0)-1):
    A = np.vstack((A,dgms0[i+1]))

#create first dimension array

def dgms1():
    k = 0
    dgms = []

    for i in masterlist:
        pattern = re.compile(r'DIMENSION 1(.*?)DIMENSION 2')
        masterlist0 = pattern.findall(i)
        dgms.append(bottleneck_format(masterlist0[0]))
        k += 1

    return dgms

dgms1 = dgms1()

#remove empty array from list of intervals
dgms1 = [i for i in dgms1 if len(i) > 0]

#combine all of the zero dimension arrays
B = dgms1[0]

for i in range(len(dgms1)-1):
    B = np.vstack((B,dgms1[i+1]))

#create second dimension array

def dgms2():

    k = 0

    dgms = []

    for i in masterlist:

        pattern = re.compile(r'DIMENSION 2(.*?)family')

        masterlist0 = pattern.findall(i)

        dgms.append(bottleneck_format(masterlist0[0]))

        k += 1

    return dgms



dgms2 = dgms2()

#remove empty array from list of intervals

dgms2 = [i for i in dgms2 if len(i) > 0]

#combine all of the zero dimension arrays

C = dgms2[0]

for i in range(len(dgms2)-1):

    C = np.vstack((C,dgms2[i+1]))



#THIS IS WHAT YOU WANT FROM THIS FILE

combined_dgms = [A,B,C]





#enter dgms into function
def int_vis(dgms):
    # seperate each dimension
    dim0 = dgms[0]
    dim1 = dgms[1]
    dim2 = dgms[2]

    #sort dim 1 by second entry and then first entry
    dim1 = list(dim1)
    dim1.sort(key = lambda x: x[1])
    dim1.sort(key = lambda x: x[0])

    #sort dim 2 by second entry and then first entry
    dim2 = list(dim2)
    dim2.sort(key = lambda x: x[1])
    dim2.sort(key = lambda x: x[0])
    print(len(dim0))
    plt.clf()

    #graph intervals
    #plt.title("Family Dimension 1")
    ylist = [i for i in range(len(dim1))]
    xlist = []

    for i in range(len(dim1)):
        #plt.hlines(y=i, xmin=dim1[i][0], xmax=dim1[i][1], linewidth=2, color='b')
        xlist.append(dim1[i][1])

    plt.plot(xlist,ylist)
    plt.xlabel(xlabel = "Intervals(Birth & Death Time)")
    plt.ylabel(ylabel = "Number of Intervals")
    plt.show()
    
    return xlist, ylist

# if __name__ == '__main__':

#     int_vis(dgms[-1])
