#!/usr/bin/env python
# coding: utf-8

# In[1]:


from family_network_model import *
from matplotlib import pyplot as plt
import networkx as nx
import ast
import pickle
from get_model_parameters import *
import ast


# In[2]:


# infinite_distances dictionary shows 'name' of all networks in data
with open('./UnionDistances/infinite_distances.txt') as f:
inf_distances = f.readline()
infinite_distances = ast.literal_eval(inf_distances)
for k in infinite_distances:
    print(k)
# infinite_distances


# In[3]:


# New file with a dictionary of other data parameters--created to make run_model.py run faster
with open('./UnionDistances/other_parameters.txt') as f:
    params = f.readline()
data_parameters = ast.literal_eval(params)
data_parameters 

# Note: data_parameters[name][0] = number of marriage edges
#       data_parameters[name][1] = probability of marriage
#       data_parameters[name][2] = probability of nonconnected marriage
#       data_parameters[name][3] = total number of nodes in network


# ## New and improved way to run model

# In[4]:


# Get data--ignore the output that says "Graph failed: ./Original_Sources/.DS_Store
                                       # Error: marriage edges from ./Original_Sources/kinsources-warao-oregraph.paj
                                       # Graph failed: ./Original_Sources/kinsources-warao-oregraph.paj"

# name of chosen network--see infinite_distances dictionary for help with names of networks (run second cell to see)
name = 'ainu_1880_as01'

# get data--number of marriage edges, probability of marriage, probability of nonconnected marriage parameters
m_e, P, NCP = get_some_parameters(name)

# get number of infinite distance unions
with open('./UnionDistances/infinite_distances.txt') as f:
    inf_distances = f.readline()
infinite_distances = ast.literal_eval(inf_distances)
# save number of infinite distance unions as a parameter
inf_dis = infinite_distances[name]

# marriage distance data of chosen network
with open('./UnionDistances/{}_distances.txt'.format(name)) as f:
    nx_dis = f.readline()
# network's marriage distances w/o infinite distances distribution
nx_distances = ast.literal_eval(nx_dis)

# number of children data of chosen network
with open('./ChildrenNumber/{}_children.txt'.format(name)) as f:
    nx_child = f.readline()
# network's number of children distribution
nx_children = ast.literal_eval(nx_child)


# In[5]:


# initialize parameters

n = 13   # n+1 = number of people in initial network
gen = 14  # gen+2 = total number of generations in network (use small number of generations for testing)
# name = 'log/' + '_test1' # change 'test' every time you run model if you want to save & differentiate the output

marriage_dist = nx_distances
children_dist = nx_children
p = P
ncp = NCP
infdis = round((inf_dis/m_e - (NCP/2))*m_e)

# run model
G, D, unions, children, infdis = human_family_network(n, gen, marriage_dist, p, ncp, infdis, children_dist, name)

# visualize resulting network
nx.draw(G, with_labels=True, font_weight='bold')


# ## Analyze Model

# In[6]:


def model_marriage_dis(unions, D):
    """Find distances of marriages formed in modeled network--used to compare to original data of distances
    Parameters:
                unions (list): list of all unions in modeled network
                D (array): matrix of distances between all nodes in modeled network
    Returns:
                distances (list): list of distances to marriage in modeled network
    
    """

    distances = []

    for pair in unions:
        p1, p2 = pair
        distances.append(D[p1][p2])
        
    return distances


# ### Visualize distribution of union distances at each generation

# In[7]:


for i in range(0, gen+2):
    print("Generation :", i)

    # read in files
#     gpickle_file = nx.read_gpickle("{}_G{}.gpickle".format(name,i))
#     D_file = np.load("{}_D{}.npy".format(name,i))
#     with open ('{}_U{}'.format(name,i), 'rb') as fup:
#         unions_file = pickle.load(fup)
#     with open('{}_C{}'.format(name,i), 'rb') as fcp:
#         children_file = pickle.load(fcp)

    # assign names 
#     G = gpickle_file
#     unions = unions_file
#     D = D_file
#     children = children_file

    # network size
    print("Number of nodes: ", len(G.nodes))

    # visualize distances
    distances = model_marriage_dis(unions, D)
    d = np.array(distances)
    mask1 = d < 100
    mask2 = d >= 100
    print("Number of marriages: ", len(unions))
    print("Number of infinite marriages: ", len(d[mask2]))
    
    plt.hist(d[mask1])
    plt.title("Generation {}".format(i))
    plt.show()

plt.hist(nx_distances, color='r')
plt.title("{} Network".format(name))
plt.show()


# ### Visualize distribution of children at each generation

# In[8]:


for i in range(1, gen+2):
    print("Generation :", i)

    # read in files
#     gpickle_file = nx.read_gpickle("{}_G{}.gpickle".format(name,i))
#     D_file = np.load("{}_D{}.npy".format(name,i))
#     with open ('{}_U{}'.format(name,i), 'rb') as fup:
#         unions_file = pickle.load(fup)
#     with open('{}_C{}'.format(name,i), 'rb') as fcp:
#         children_file = pickle.load(fcp)

    # assign names 
#     G = gpickle_file
#     unions = unions_file
#     D = D_file
#     children = children_file

    # network size
    print("Number of nodes: ", len(G.nodes))

    plt.hist(children, color='g', bins=6)
    plt.title("Generation {}".format(i))
    plt.show()

plt.hist(nx_children, color='r', bins=6)
plt.title("{} Network".format(name))
plt.show()


# In[9]:


print(len(unions))


# In[10]:


print(unions)


# ### Visualize largest connected component of network at each generation

# In[11]:


# Visualize largest connected component

for i in range(gen+2):
    print("Generation :", i)

    # read in file
#     gpickle_file = nx.read_gpickle("{}_G{}.gpickle".format(name,i))
#     G = gpickle_file
    
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    nx.draw(G0, node_size=15)
    plt.show()


# In[ ]:




