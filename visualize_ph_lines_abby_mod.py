from matplotlib import pyplot as plt
import numpy as np
import time
import networkx as nx
import scipy as sp
from ripser import ripser

'''MAKES THE PERSISTENCE SURFACES.'''

#enter dgms into function
def int_vis(dgms,dim=1):
    
#     if type(dgms) == list:
#         for d in dgms:
#             int_vis(d)
#     else:
        
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
        plt.clf()

        #graph intervals
        if dim == 1:
            plt.title("Family Dimension 1")
            for i in range(len(dim1)):
                plt.hlines(y=i, xmin=dim1[i][0], xmax=dim1[i][1], linewidth=2)#, alpha = .3)
                
        else:
            plt.title("Family Dimension 2")
            for i in range(len(dim2)):
                plt.hlines(y=i, xmin=dim2[i][0], xmax=dim2[i][1], linewidth=2)#, alpha = .3)
    
#         plt.yticks(np.arange(0,len(dim1)))
        plt.xlabel(xlabel = "Intervals(Birth & Death Time)")
        plt.ylabel(ylabel = "Number of Intervals")
        plt.show()
        
def int_vis2(dgms,dim):
#     if type(dgms) == list:
#         for d in dgms:
#             int_vis(d)
#     else:
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
        plt.clf()
        #graph intervals
        if dim == 0:
            plt.title("Family Dimension 0")
            for i in range(len(dim0)):
                plt.hlines(y=i, xmin=dim0[i][0], xmax=dim0[i][1], linewidth=2)#, alpha = .3)
        if dim == 1:
            plt.title("Family Dimension 1")
            for i in range(len(dim1)):
                plt.hlines(y=i, xmin=dim1[i][0], xmax=dim1[i][1], linewidth=2)#, alpha = .3)
        if dim == 2:
            plt.title("Family Dimension 2")
            for i in range(len(dim2)):
                plt.hlines(y=i, xmin=dim2[i][0], xmax=dim2[i][1], linewidth=2)#, alpha = .3)
#         plt.yticks(np.arange(0,len(dim1)))
        plt.xlabel(xlabel = "Intervals(Birth & Death Time)")
        plt.ylabel(ylabel = "Number of Intervals")
        plt.show()

def step(dgms, dim):
    x = [1,2,3,4]
    y = [5, 5,4, 3]
    plt.step(x, y)
    plt.show()


    if dim ==0:
        dim = dgms[0]
    elif dim ==1:
        dim1 = dgms[1]
    else:
        dim2 = dgms[2]
    interval = []
    count = []
    if len(dim) != 0:
        dim = [str(i) for i in dim]
        unique = np.unique(dim)
        for ele in unique:
            interval.append(ele)
            count.append(dim.count(ele))
    else:
        print("There are no intervals in this dimension")
        
def vis3d(dgms,dim,noisy=False,show=False):
    dim0 = []
    dim1 = []
    dim2 = []
    
    for dgm in dgms:
        # seperate each dimension
        dim0.append(dgm[0])
        dim1.append(dgm[1])
        dim2.append(dgm[2])
        #sort dim 1 by second entry and then first entry
        dim1[-1] = list(dim1[-1])
        dim1[-1].sort(key = lambda x: x[1])
        dim1[-1].sort(key = lambda x: x[0])
        #sort dim 2 by second entry and then first entry
        dim2[-1] = list(dim2[-1])
        dim2[-1].sort(key = lambda x: x[1])
        dim2[-1].sort(key = lambda x: x[0])
        plt.clf()
    
    y = []
    xmin = []
    xmax = []
    dim_to_plot = [dim0,dim1,dim2][dim]
    len_check = []
    for j in range(len(dim_to_plot)):
        smin = []
        smax = []
        sy = []
        
        #Check the length of each dimension
        len_check.append(len(dim_to_plot[j]))
        for i in range(len(dim_to_plot[j])):
            sy.append(i)
            a = dim_to_plot[j][i][0]
            b = dim_to_plot[j][i][1]
            smin.append(a)
            smax.append(b)
        y.append(sy)
        xmin.append(smin)
        xmax.append(smax)
        
    if np.allclose(len_check,np.zeros_like(len_check)):
        print('No information to plot from dimension', str(dim)+'.')
    else:
        
        base_y = max([len(x) for x in y])
        R = len(y)  

        for i in xmax:
            while len(i) < base_y+1:
                i.append(0)
        for i in xmin:
            while len(i) < base_y+1:
                i.append(0)
        Zmax = np.array(xmax)
        if type(Zmax[0]) != np.ndarray:
            Zmax = np.array([np.array(x) for x in Zmax])
        Zmin = np.array(xmin)
        if type(Zmin[0]) != np.ndarray:
            Zmin = np.array([np.array(x) for x in Zmin])
        X,Y = np.meshgrid(np.arange(base_y+1),np.arange(R))


        if show:
            fig = plt.figure()
            ax=fig.add_subplot(111,projection='3d')
            ax.plot_surface(X,Y,Zmax,color='mediumturquoise',alpha=.9,label='Death')
            ax.plot_surface(X,Y,Zmin,color='purple',alpha=.6,label='Birth')
            ax.set_xlabel('Interval Number')
            ax.set_ylabel('Node/Radius')
            ax.set_zlabel('PH Interval')
    #         ax.legend()
        
            plt.show()

        return X,Y,Zmax,Zmin


def write_better(results):
    dims = []
    for d in results:
        if 0 not in d.shape:
            dims.append(np.unique(d,axis=0,return_counts=True)[:-1])
        else:
            dims.append(None)
    return dims

def Ripser(G,t=False):
    start = time.time()
    distance = nx.floyd_warshall_numpy(G, nodelist=None, weight='weight')
    distance = sp.sparse.csr_matrix(distance)
    results = ripser(distance,distance_matrix=True,maxdim = 2)["dgms"]
    if t == True:
        print("Total Time:",time.time()-start)
    return results