import networkx as nx
import random
import functools
import operator
import numpy as np
import ast
from scipy import stats
from sklearn.neighbors import KernelDensity as KDE
from scipy import interpolate
import itertools
import pickle

#%%
def add_marriage_edges(all_fam, all_unions, D, marriage_probs, p, ncp, n, infdis):
    """Add marriage edges to a given generation
    Paramters:
                all_fam (list): list of all nodes in current generation--all eligible nodes for marrige. 
                                (Flattened list of families)
                D (array): nxn matrix of distances between nodes
                marriage_probs (dict): dictionary of marriage distance probabilities. 
                                        (e.g., value at key 5 is probability of distance of length 5)
                p (float): probability of marrying
                ncp (float): probability of a nonconnected marriage
                n (int): current number of nodes in network
                inf_dis (int): number of marriage pairs in data that have an infinite distance
    Returns: 
                new_unions (list): list of edges that represent new unions between nodes
                no_unions (list): list of nodes with no unions
                all_unions (list): list of all unions since generation 0
                n (int): current number of nodes in network
                m (int): number of nonconnected nodes added to network
    
    """
    unions = []
    # number of non-connected people to add
    m = round(((ncp*len(all_fam))/(1-ncp))/2)
    infdis -= m
    # list of people not connected to family network
    nc_ppl = []
    for i in range(1, m+1):
        nc_ppl.append(n+i)
    # update number of people in network
    n = n + m
    # marry off non-connected people
    for nc in nc_ppl:
        spouse = random.choice(all_fam)
        unions.append((nc, spouse))
        all_fam.remove(spouse)
    
    # select how many of the n+m people get married based on probability of marriage
    k = round(((n+1)*p)/2)
    
    # get all possible pairs of rest of nodes
    all_pairs = list(itertools.combinations(all_fam, 2))

    # get all possible pairs of nodes that can marry
    poss_dis = {}
    for pair in all_pairs.copy():
        # get distance of pair
        dis = D[pair[0]][pair[1]]
        
        # check that pair can marry (i.e., siblings can't marry)
        if dis < min(marriage_probs):
            all_pairs.remove(pair)   

        # keep track of distances of all possible nodes to marry
        else:
            if dis >= 100:   
                poss_dis[pair] = 100  # all infinite distances are '100'
            else:
                poss_dis[pair] = dis



    while (len(all_pairs) > 0) and (len(unions) <= k):   
        # find probabilities of all possible distances--must update after creating each union
        dis_probs = []
        for d in poss_dis.values():
            dis_probs.append(marriage_probs[d])

        # normalize probabilities--must update after creating each union
        rel_probs = list(np.array(dis_probs)/sum(dis_probs))  # relative probability of distances

        # choose pair to marry based on relative probability of distances
        marry = random.choices(population=all_pairs, weights=rel_probs)[0]
        unions.append(marry)

        # remove all pairs that include one of the nodes of 'marry'
        for pair in all_pairs.copy():
            if (pair[0] == marry[0]) or (pair[0] == marry[1]) or (pair[1] == marry[0]) or (pair[1] == marry[1]):
                all_pairs.remove(pair)
                poss_dis.pop(pair)
        
    # keep track of nodes that didn't marry 
    no_unions = list(set(all_fam) - set(functools.reduce(operator.iconcat, unions, ())))
    all_unions += unions
    
    return unions, no_unions, all_unions, n, m, infdis

#%%
def add_children_edges(unions, n, child_probs, all_children):
    """Add children edges to a given generation of unions
    Parameters: unions (list): unions formed in current generation
                n (int): current number of nodes in graph
                child_probs (dict): dictionary of number of children probabilities
                                    (e.g., value at key 5 is probability of having 5 children)
    Returns: child_edges (list): list of new parent-child edges to add to graph
             families (list): list of lists where each inner list represents siblings 
             n (int): new number of nodes in graph after adding children
             all_children (list): list of number of children per union since generation 0
    
    """
    
    families = []

    child_edges = []
    
    for union in unions:
        # how many children the union will have--based on children data distribution
        num = random.choices(population=list(child_probs.keys()), weights=list(child_probs.values()))[0]

        # add children
        if num != 0:
            # initialize list of children
            children = []
            # add children nodes to graph
            for c in range(num):
                # specify node number (label) to add
                n = n+1
                children.append(n)
                # add edges from child to parents
                child_edges.append((union[0], n))
                child_edges.append((union[1], n))
            # keep track of families
            families.append(children)

        # no children so add empty list
        else:
            families.append([])
        
        all_children.append(num)

    return child_edges, families, n, all_children

#%%
def update_distances(D, n, unions, no_unions, families):
    """Build a distance matrix that keeps track of how far away each node is from each other.
        Need to update distances after new nodes added to graph (i.e., after adding children)
    Parameters: D (array): "old" matrix of distances
                n (int): number of nodes currently in graph
                unions (list):
                no_unions (list):
                families (list):
    Returns: D1 (array): "new" (updated) matrix of distances
    """
    
    # initialize new matrix
    D1 = np.zeros((n+1,n+1))
    
    # number of old nodes (grandparent+)
    oldies = np.shape(D)[0] - ((len(unions)*2) + len(no_unions))

    # fill in new matrix with old information
    oldn = np.shape(D)[0]
    for i in range(oldn):
        for j in range(oldn):
            D1[i][j] = D[i][j]
            D1[j][i] = D[j][i]
            
    # compute new distances        
    for u, union in enumerate(unions):
        u_children = families[u]
        
        for other in unions:
            if (union != other):
                o_children = families[unions.index(other)]
            
                # find all possible distances from union to other
                d1 = D[union[0]][other[0]]
                d2 = D[union[1]][other[0]]
                d3 = D[union[0]][other[1]]
                d4 = D[union[1]][other[1]]

                # compute distance between children of union and other
                p0_d = min(d1, d2) + 1     # distance from child to other[0]
                p1_d = min(d3, d4) + 1     # distance from child to other[1]
                
                # compute distance between children of union and children of other
                d = min(d1, d2, d3, d4) + 2

                # add distances to matrix
                for uc in u_children:
                    D1[uc][other[0]] = p0_d
                    D1[other[0]][uc] = p0_d

                    D1[uc][other[1]] = p1_d
                    D1[other[1]][uc] = p1_d
                
                    for oc in o_children:
                        D1[uc][oc] = d
                        D1[oc][uc] = d
                        
                        
        # add immediate family distances
        for ch in u_children:
            # add sibling distances
            for c in u_children:
                if ch != c:
                    D1[ch][c] = 2
                    D1[c][ch] = 2
            # add parent-child distances
            D1[ch][union[0]] = 1
            D1[union[0]][ch] = 1
            D1[ch][union[1]] = 1
            D1[union[1]][ch] = 1

        # get distances of nonmarried nodes to union
        for nm in no_unions:
            d5 = D[nm][union[0]]
            d6 = D[nm][union[1]]
            # add distance of nonmarried nodes to children
            for child in u_children:
                D1[nm][child] = min(d5, d6) + 1
                D1[child][nm] = min(d5, d6) + 1
                
        # need to find shortest path from parents to all "old" nodes      
        for old in range(oldies):
            d7 = D[old][union[0]]
            d8 = D[old][union[1]]
            # children inherit parent distance
            for child in u_children:
                D1[old][child] = min(d7, d8) + 1
                D1[child][old] = min(d7, d8) + 1

    return D1

#%%
def toKDE(data, bandwidth, kernel='gaussian'):
    #data is a list of numbers that occur, and you want to find a KDE of their frequency.
    return KDE(kernel=kernel,bandwidth=bandwidth).fit(np.array(data).reshape(-1,1))

#%%
def get_probabilities(data, gen=0):
    """Create dictionary of probabilities based on real data distribution
    Parameters: 
                data (list): data from a real family network
                gen (int): number of generations the network will grow (only need for marriage distribution)         
    Returns: 
                probs (dictionary): probabilities of all possible datapoints
                                    (e.g., probability at index 5 is probability of distance/children of length 5)
    
    """
    
    # changing bandwidth changes KDE
    bandwidth = 1
    
    # get kernel density of data
    kde = toKDE(data, bandwidth)
    x = np.arange(min(data)-1, max(data)+2, 1) # start and stop might need to change
    domain = x[:,np.newaxis] 
    logs = kde.score_samples(domain)  # evaluate log density model on data
    y = np.exp(logs)  # get density model

    # fit spline (i.e., fit equation to density curve to then be able to integrate)
    spl = interpolate.InterpolatedUnivariateSpline(x,y)
    
    # create dictionary of probabilities by integrating under density curve
    probs = {}
    keys = set(data)
    for i in range(min(keys), max(keys)+gen+2):
        probs[i] = spl.integral(i-.5, i+5)

    return probs

#%%
def human_family_network(n, gen, marriage_dist, p, ncp, infdis, children_dist, name):
    """Create an artifical human family network
    Parameters: 
                n (int): number of nodes for initial graph minus 1
                          (e.g., to initialize a graph of 100 nodes, set n=99)
                gen (int): number of generations for network to grow for gen+1 total generations in network
                          (e.g., when gen=3, the network grows from generation 0 to generation 4)
                marriage_dist (list): data distribution from real family network of how "far" people married
                p (float): probability of marriage
                ncp (float): probability of marriage of nonconnected nodes
                infdis (int): number of infinite distance marriages
                children_dist (list): data distribution from a real family network of number of children per union
                name (str): name for prefix of saved files
    
    Returns: G (graph): contrived human family network
             D (array): matrix of distances between nodes in network
             all_unions (list): list of all marriage pairs in network
             all_children (list): list of number of children per union since generation 0
    
    """

    # generate empty graph
    G = nx.Graph()

    # add nodes to graph
    G.add_nodes_from([i for i in range(n+1)])

    # create lists needed for function
    families = []
    for node in G.nodes():
        families.append([node])
    all_fam = functools.reduce(operator.iconcat, families, [])
    all_unions = []
    all_children = []
    
    
    # initialize distance matrix of specified size
    D = np.zeros((n+1,n+1))
    
    # create list of all possible distances (including infinite distances)
    all_distances = marriage_dist.copy() + [100 for i in range(infdis)]   # best number of infdis to add?

    # fill in distances with data
    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                # distance from self to self
                D[i][j] = 0
            else:
                d = random.choice(marriage_dist)  # using marriage_dist instead of all_distances gives D all finite distances
                D[i][j] = d
                D[j][i] = d
                
    
    
    # get probabilities of possible distances to use in marriage function
    marriage_probs = get_probabilities(marriage_dist, gen=gen) # dictionary of probabilities of all finite distances
    marriage_probs[100] = (infdis/len(all_distances))/2  # include probability of infinite distance
    factor = 1.0/sum(marriage_probs.values())   # normalizing factor
    # normalize values for finite and infinite distances
    for k in marriage_probs:
        marriage_probs[k] = marriage_probs[k]*factor
    
    # get probabilities of possible number of children to use in children function
    child_probs = get_probabilities(children_dist)  # dictionary of probabilities of all possible number of children
    
    
    # add specified number of generations to network
    for i in range(gen+1):
        print('generation: ', i)
        
        # save output at each generation
        Gname = "{}_G{}.gpickle".format(name, i)   # save graph
        nx.write_gpickle(G, Gname)
        Dname = "{}_D{}.npy".format(name, i)   # save D
        np.save(Dname, D)
        Uname = "{}_U{}".format(name, i)   # save unions
        with open(Uname, 'wb') as fup:
            pickle.dump(all_unions, fup) 
        Cname = "{}_C{}".format(name, i)   # save children
        with open(Cname, 'wb') as fcp:
            pickle.dump(all_children, fcp)   
        
        # create unions between nodes to create next generation
        unions, no_unions, all_unions, n, m, infdis = add_marriage_edges(all_fam, all_unions, D, marriage_probs, p, ncp, n, infdis)
        G.add_edges_from(unions)
  
        oldn = n-m
     
        for j in range(m):
            # add non-connected ppl to distance matrix--infinte distances with everyone else
            r = np.ones((1,oldn+2+j))*100
            r[0,-1] = 0  # distance to self is 0
            c = np.ones((oldn+1+j,1))*100
            D = np.hstack((D, c))
            D = np.vstack((D, r))
  
        # add children to each union
        children, families, n, all_children = add_children_edges(unions, n, child_probs, all_children)
        G.add_edges_from(children) 
        all_fam = functools.reduce(operator.iconcat, families, [])
        
        # update distances between nodes
        D = update_distances(D, n, unions, no_unions, families)
        
            
        # save output of last generation
        if i == gen:
            print("Last generation: ", i+1)
            Gname = "{}_G{}.gpickle".format(name, i+1)   # save graph
            nx.write_gpickle(G, Gname)
            Dname = "{}_D{}.npy".format(name, i+1)   # save D
            np.save(Dname, D)
            Uname = "{}_U{}".format(name, i+1)   # save unions
            with open(Uname, 'wb') as fup:
                pickle.dump(all_unions, fup) 
            Cname = "{}_C{}".format(name, i+1)   # save children
            with open(Cname, 'wb') as fcp:
                pickle.dump(all_children, fcp)
    
    print(G.number_of_nodes())
        
    return G, D, all_unions, all_children, infdis