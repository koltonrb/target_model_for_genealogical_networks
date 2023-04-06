import networkx as nx
#import random
#import functools
#import operator
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ast
#from scipy import stats

from scipy.stats import gaussian_kde
#from sklearn.neighbors import KernelDensity as KDE
#from scipy import interpolate

import itertools
import pickle
#from time import time
#from functools import wraps
import os
import regex as re
from write_model_to_pajek_ore_graph import format_as_pajek
from itertools import combinations, product
#%%
def makeOutputDirectory(out_directory, name):
    """
    Make an output directory to keep things cleaner

    Returns a full output path to the new directory
    """
    ver = 1
    output_dir = os.path.join(out_directory, name + '_')
    while os.path.exists(output_dir+str(ver)):
        ver += 1
    output_dir += str(ver)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


#%%
def find_file_version_number(out_directory, filename, extension):
    ver = 1
    output_dir = os.path.join(out_directory, filename + '_')
    while os.path.exists(output_dir+str(ver)+extension):
        ver += 1
    filename = filename + '_' + str(ver)
    return filename



#%%
def get_graph_path(name, path='./Original_Sources/'):
    """
    PARAMETERS:
        name: (str) the name of the kinsources data set (see below for format)
    RETURNS:
        path: (str) path to directory prepended to the full name of specified
              kinsources file
    """
    return path + 'kinsources-'+name+'-oregraph.paj'


#%%
def get_num_people(name):
    """
    gets the number of people (total number of verticies) in the .paj file
    This function assumes that the number of nodes is correctly reported in the
    3rd line (index 2) of the .paj file. For example, contents should begin in
    the following format (defined in the funtion below):
        ['*Network Ore graph Tikopia.puc\n',
         '\n',
         '*vertices 294\n',
         "1 'X (1)' ellipse\n",
         ...]

    PARAMETERS:
        name: (str) the name of the kinsources data set
    RETURNS:
        num_people: (int) total number of people in the graph
    """
    path_to_graph = get_graph_path(name)
    # open and read graph file
    with open(path_to_graph, 'r') as file:
        contents = file.readlines()

    num_people = contents[2]
    num_people_pattern = re.compile("[0-9]+")
    num_people = int(num_people_pattern.findall(num_people)[0])

    return num_people


#%%
def get_graph_stats(name, distance_path='./Kolton_distances/', child_number_path='./ChildrenNumber/'):
    """
    Gets the statistics of a specified kinsources dataset
    PARAMETERS:
        name: (str) the name of the kinsources data set
        distance_path: (str) the filepath to the directory containing the saved
            text files containing the distance to marriage distributions (the output
            of timing_kolton_distance_algorithm.py)
        child_number_path: (str) the filepath to the directory containing the
            saved text files containing the children per couple distributions
    RETURNS:
        marriage_dists: (list of int) one entry per marriage indicating how many
            generations between spouses (reported in the number of parent-child
            edges crossed so that distance between siblings is 2) in the
            specified dataset.  If no common ancestor (IE an infinite distance)
            then the corresponding entry is -1
        num_marriages: (int) total number of marriage edges in the specified
            dataset
        prob_inf_marriage: (float) number of infinite marraiges divided by total
            number of marriages in the specified dataset
        prob_finite_marriage: (float) number of non-infinite marriages divided
            by total number of marriages in the specified dataset
        children_dist: (list of int) one entry per pair of parents, indicating
            how many child edges each parent in the couple share
        num_people: (int) total number of nodes in the graph named.
    """
    with open(distance_path + '{}.txt'.format(name)) as infile:
        marriage_dists, num_inf_marriages, fraction_inf_marriage = [ast.literal_eval(k.strip()) for k in infile.readlines()]

    # number of children data of chosen network
    with open(child_number_path + '{}_children.txt'.format(name)) as f:
        nx_child = f.readline()
    children_dist = ast.literal_eval(nx_child)

    num_marriages = len(marriage_dists)
    num_people = get_num_people(name)
    prob_marriage = num_marriages * 2 / num_people  # *2 since 2 spouses per marriage
    prob_inf_marriage = prob_marriage * fraction_inf_marriage
    prob_finite_marriage = prob_marriage - prob_inf_marriage

    return marriage_dists, num_marriages, prob_inf_marriage, prob_finite_marriage, children_dist, num_people


#%%
def get_probabilities(data, bandwidth=1, is_child=False, eps=1e-7):
    """
    given a list (either of distances to marriage for a marriage distribution or
    numbers of children for a children distribution), get_probabilities() produces
    a dictionary of probabilities. NOTE: the resulting "probabilities" should
    not be expected to sum to 1.  If a true probability distribution is desired
    then you should normalize the resulting distribution.  The resulting
    dictionary has entries beyond the data supplied (for example if a supplied
    marriage distribution has a maximum distance of 14 the resulting dictionary
    has entries for distances greater than 14 to allow us to use the
    datastructure without key errors should a larger number be drawn; we add
    1000 entries beyond the maximum.  If ever more than 1000 generations are to
    be run in the model, then this function should be modified).
    PARAMETERS:
        data (list): data taken from an actual family network
        bandwidth (int):  used as an argument in to_KDE(), the std deviation of
            each kernel in the sum (see documentation)
    RETURNS:
        probs (dictionary): keys are the entries of data and successive values,
            too (we lengthen the right tail of the distribution).
    """
    # # ??? should this data list NOT contain the infinite entries?
    # #     currently includes infinite entries (under key 999)
    # #     would it be better to have infinity represented as a distance of 0?
    # data = np.array(data)
    # data = data[data > -1]  # only use the non-infinite distances
    # kde = to_KDE(data, bandwidth)
    # domain = np.arange(min(data)-1, max(data)+1000, 1)  # ??? shouldn't I go from 0 to inf or from 2 to inf all the time?
    # domain = domain[:, np.newaxis]
    # logs = kde.score_samples(domain)
    # y = np.exp(logs)
    #
    # # fit spline (IE fit equation to density curve to then be able to integrate)
    # spl = interpolate.InterpolatedUnivariateSpline(domain, y)
    #
    # # create a dictionary of probabilities by integrating the density curve
    # probs = {i:spl.integral(i-0.5, i+0.5) for i in range(min(data), max(data)+1000)}
    # return probs
    data = np.array(data)
    data = data[data > -1]  # only use the non-infinite distances
    if is_child:  # IE is child
        domain = np.arange(0, max(data), 1)
    else:
        # ??? I feel like marriage distances should always have all distances
        #     possible, even the gross ones (IE we need to count from 2 not the
        #     minimum distance seen in the dataset)
        # domain = np.arange(2, max(data)+1000, 1)
        domain = np.arange(min(data), max(data)+1000, 1)  # ??? shouldn't I go from 0 to inf or from 2 to inf all the time?

    kde = gaussian_kde(data, bw_method=bandwidth)
    # probs = {x:kde2.integrate_box_1d(-np.inf, x) for x in domain}  # CDF, discretized
    probs = {x:kde.integrate_box_1d(x-0.5, x+0.5) for x in domain}
    
    if not is_child:
        eps = min([k for k in probs.values() if k != 0]) / 2
        probs = {key:val + eps for key, val in probs.items()}
    return probs

#%%

def get_difference_in_probabilities(target_probs, current, eps=1e-6):
    """
    This method accepts both the target marriage distribution AND the 
    current-state model marriage distribution.  It will subtract the 
    current-state from the target probabilites, flooring at some positive 
    epsilon.  The returned probabiltiy distribution will then favor any 
    marriages of those distances which have not yet been drawn in proportion 
    with the target distribution's frequency for that distance
    
    PARAMETERS:
        target_probs (dictionary): keys are marriage distances, values are 
            probabilities.  This is the result of 
            get_probabilities(marriage_dist) (called finite_marriage_probs 
            below). This should already be normalized.  
        current (list): list of marriage distances currently represented in the
            graph.  
    """
    current_probs= get_probabilities(current)
    current_probs = {key:value/sum(current_probs.values()) for key, value in zip(current_probs.keys(), current_probs.values())} # normalize 
    # need every key that occurs in target to also occur in current_probs
    current_probs = current_probs | {key : 0 for key in target_probs if key not in current_probs}
    
    adjusted_probs = {key: target_probs[key] - current_probs[key] if target_probs[key] - current_probs[key] >= 0  else eps for key in target_probs.keys()}
    
    # normalize 
    adjusted_probs = {key:value/sum(adjusted_probs.values()) for key, value in zip(adjusted_probs.keys(), adjusted_probs.values())} # normalize 
    
    return adjusted_probs 
    
    
# TODO: it would be better to make this accept marriage_probs as an argument rather than
#       using marraige_probs as a global argument, right?
# TODO: need to make it so that the model runs, saves, and exits with an error message if people should ever be an empty list
# people=generation_of_people
# prev_people=prev_generation_still_single
# prob_marry_immigrant = prob_inf_marriage
# prob_marry = prob_finite_marriage

#%%
# target_prob_inf = prob_inf_marriage
# target_prob_finite = prob_finite_marriage 
# current_marriage_dist_list = all_marriage_distances  
# current_total_num_single = total_num_single 
#%%
def get_difference_in_types_of_marriage(target_prob_inf, target_prob_finite, current_marriage_dist_list, verbose=False):
    """
    This function does not adjust the total probability of marriage 
    (prob_finite + prob_inf remains the same).  It does adjust the mixture of 
    inf and finite probabilities.
    
    PARAMETERS:
        target_prob_inf (float): probability that a NODE in the target network 
            takes part in an infinite-distance marriage. NOTE: should be
            prob_inf_marriage as returned by get_graph_stats() so that 
            target_prob_inf + target_prob_finite = 1.0 - target_prob_single
        target_prob_finite (float): probability that a NODE in the target 
            network takes part in an infinite-distance marriage. NOTE: should 
            be prob_finite_marriage as returned by get_graph_stats() so that 
            target_prob_inf + target_prob_finite = 1.0 - target_prob_single
        current_marriage_dist_list (list of int): list of all distances of all 
            marriages currently in the modeled graph G.  Must follow the 
            convention employed throughout that a recorded distance of -1 
            indicates an infinite-distance marriage (partners share no common 
            ancestor). 
            
    RETURNS:
        new_prob_inf (float): the sum of target_prob_inf and the difference 
            between the probability of an infinite distance marriage in the 
            target graph and the proportion of infinite distance marriages in 
            the current state model graph (scaled by the target graph's 
            probability of marriage) 
        new_prob_finite (float): the sum of target_prob_finte and the 
            difference between teh probability of a finite distance marriage in 
            the target graph and the proportion of finite distance marriages in 
            the current state model graph (scaled by the target graph's 
            probability of marriage)
    """
    
    target_prob_marry = target_prob_inf + target_prob_finite  # complement of this is target_prob_single
    
    current_marriage_dist_list = np.array(current_marriage_dist_list)
    current_proportion_inf = sum(current_marriage_dist_list == -1) / len(current_marriage_dist_list)  # percentage of marriages, not percentage of people at inf distance
    current_proportion_finite = sum(current_marriage_dist_list != -1) / len(current_marriage_dist_list)
    
    current_prob_inf    = target_prob_marry * current_proportion_inf  # scale by proportion of target population that marries
    current_prob_finite = target_prob_marry * current_proportion_finite 
    
    # diff_inf = -1 * diff_finite
    diff_inf = target_prob_inf - current_prob_inf 
    diff_finite = target_prob_finite - current_prob_finite
    
    new_prob_inf = max(target_prob_inf + diff_inf, 0)
    new_prob_finite = max(target_prob_finite + diff_finite, 0) 
    
    if new_prob_inf == 0:
        new_prob_finite = target_prob_marry 
    if new_prob_finite == 0:
        new_prob_inf = target_prob_marry
    if verbose:
        if new_prob_inf < 0 or new_prob_finite < 0:
            print('current_proportion_inf', current_proportion_inf)
            print(' current_proportion_finite', current_proportion_finite)
            print('current_prob_inf', current_prob_inf)
            print(' current_prob_finite', current_prob_finite)
            print('diff_inf', diff_inf)
            print('diff_finite', diff_finite)
            print('new_prob_inf', new_prob_inf)
            print(' new_prob_finite', new_prob_finite)
    return new_prob_inf, new_prob_finite

    # # this will find prob marry in a way that will sum to one with prob single
    # current_prob_marry = len(current_marriage_dist_list) * 2 / (len(current_marriage_dist_list) * 2 + current_total_num_single)
    # current_marriage_dist_list = np.array(current_marriage_dist_list)
    # current_prob_inf = sum(current_marriage_dist_list == -1) / len(current_marriage_dist_list) * current_prob_marry       # x/100 * marry% = new fraction of marriages that are inf dist
    # current_prob_finite = sum(current_marriage_dist_list != -1) /  len(current_marriage_dist_list)  * current_prob_marry  # makes it so prob_inf + prob_finite + prob_single = 1
    # target_proportion_inf    = target_prob_inf    / target_prob_marry 
    # target_proportion_finite = target_prob_finite / target_prob_marry 
    
    # # if current_prob_inf_marriage > target_prob_inf: 
    # #     # too many inf marriages in model
    # #     new_prob_inf = min(current_prob_inf - target_prob_inf, target_prob_inf/2)
    # #     new_prob_finite = target_prob_finite + (target_prob_inf - new_prob_inf)
    # # else:
    # #     # not enough inf marriages in model
    # #     # so current_prob_finte > target_prob_finite
    # #     new_prob_finite = target_prob_finite - current_prob_finite
    # #     new_prob_inf = current_prob_inf + (target_prob_)
    
    
    # return max(0, target_prob)
    

#%%
# people = generation_of_people
# prev_people = prev_generation_still_single
# finite_marriage_probs = new_finite_marriage_probs
# prob_marry_immigrant = prob_inf_marriage
# prob_marry = prob_finite_marriage
#%%
def kolton_add_marriage_edges(people, prev_people, num_people, finite_marriage_probs, prob_marry_immigrant, prob_marry, D, indices, tol=0):
    """
    Forms both infinite and finite distance marriages in the current generation
    PARAMETERS:
        people:  (list) of the current generation (IE those people elligible for
                marriage)
        prev_people: (list) of those in the previous generation who are yet unmarried.
        num_people: (int) the number of nodes/people currently in the graph
        finite_marriage_probs: (dictionary) keys are marriage distances, values
            are probabilities.  Note that this dictionary should only include
            entries for NON-inifite marriage distances, should have
            non-negaitive values which sum to 1, and should have a long right
            tail (IE lots of entries which map high (beyond what occurs in the
            example dataset) distances to zero(ish) probabilties)
        prob_marry_immigrant: (float) the probablility that a given node will marry
                a immigrant (herein a person from outside the genealogical network,
                without comon ancestor and therefore at distance infinity from the
                nodes in the list 'people') (formerly 'ncp')
        prob_marry: (float) the probability that a given node will marry another
                node in people
        D: ((len(people) x len(people)) numpy array) indexed array of distance
            between nodes in people (siblings are distance 2)
        indices: (dictionary) maps node name (int) to index number in D (int)
    RETURNS:
        unions: (list of tuples of int) marriages formed.  Entries are of two
            types: 1) infite distance marraiges: one spouse is selected
            uniformly at random from the community (people) while the other is
            an immigrant to the community (IE a new node NOT listed in people).
            2) finite distance couples: both spouses are members of the
            community (IE listed in people).  These couples are selected at
            random according to the marriage_probs)
        num_immigrants: (int) the number of NEW people added to the graph.  As
            implemented ALL new people get married to someone in the current
            generation (herein people)
        marriage_distances: (list of int) one entry per marriage created during
            this function call (IE within this generation).  Each entry
            indicates the distance between spouses through their nearest common
            ancestor.  As before, a distnace of -1 indicates an infinite
            distance (IE one spouse immigrated into the community)
        wont_marry: (list) of nodes in people who did not marry, will attempt
            to marry someone from the next generation
    """
    marriage_distances = []

    people_set = set(people)  # for fast removal later
    # find the next 'name' to add to your set of people
    next_person = num_people + 1
    # number of non-connected people to add
    num_immigrants = round(prob_marry_immigrant * len(people))  # m
    num_couples_to_marry = round(prob_marry * len(people) / 2)
    # marry off the immigrants at random to nodes in the current generation
    immigrants = [k for k in range(next_person, next_person + num_immigrants)]
    marry_strangers = np.random.choice(people, size=num_immigrants, replace=False)
    unions = {(spouse, immigrant) for spouse, immigrant
                                    in zip(marry_strangers,
                                           immigrants)}
    # remove the married people from the pool #2ndManifesto
    people_set = people_set - set(marry_strangers)
    # and record the (infinite) distances of each marriage
    marriage_distances += [-1 for k in range(num_immigrants)]

    # now that you've married off some fraction of the generation to new nodes (ie to immigrants)
    # divide the current generation into two camps, those who will marry among this and the previous generation
    # AND those who will marry next generation
    will_marry = set(np.random.choice(list(people_set), size=len(people_set)//2, replace=False))
    wont_marry_until_next_time = [node for node in people_set if node not in will_marry]  # won't attempt to form marriages until next generation
    # add in singles from the previous generation
    people_set = will_marry | set(prev_people)
    # get number of people to marry
    # num_couples_to_marry = round(len(people_set)*prob_marry/2)  # this line grabs a fraction of those who either stay single or marry at a finite difference but it doesn't account that part of that gen already married strangers
    # # num_couples_to_marry = round((len(people_set) + len(marry_strangers)) * prob_marry / 2)
    # get all possible pairs of the still single nodes
    # rejecting possible parrings which have a common ancestor more recently
    # than allowed by finite_marriage_probs (IE this is where we account that siblings
    # don't marry in most cultures (but still can such as in the tikopia_1930
    # family network))
    possible_couples = [(man, woman) for man, woman in itertools.combinations(people_set, 2)]
    last_gen_couples = [(man, woman) for man, woman in itertools.combinations(prev_people, 2)]
    # we want combinations of couples from those in will_marry and prev_people (singles from last generation)
    # but not pairings where boths spouses are in the previous generation
    possible_couples = set(possible_couples) - set(last_gen_couples)

    possible_couples = {(man, woman): D[indices[man]][indices[woman]]
                        for man, woman in itertools.combinations(people_set, 2)
                        if D[indices[man]][indices[woman]] >= min(finite_marriage_probs)}
    iter = 0
    print("len possible couples: ", len(possible_couples))
    while possible_couples and iter < num_couples_to_marry:
        # find the probabilities of all possible distances
        # must update after each marriage
        # change to a data structure suited to random draws:
        # FIXME will this preserve ordering between keys and values?
        
        print('possible_couples', possible_couples)
        possible_couples_array = np.array(list(possible_couples.keys()))
        print('possible couples_array', possible_couples_array)
        dis_probs = np.array([finite_marriage_probs[d] for d in possible_couples.values()])
        print('dis_probs 1', dis_probs)
        dis_probs[np.abs(dis_probs) < tol] = 0  # prevent "negative zeros"
        print('dis_probs 2', dis_probs)
        dis_probs = dis_probs / np.sum(dis_probs)  # normalize
    
        print('dis_probs 3', dis_probs)
        # choose couple based on relative probability of distances
        couple_index = np.random.choice(np.arange(len(possible_couples)), p=dis_probs)
        couple = possible_couples_array[couple_index]
        unions.add(tuple(couple))
        # and save the distance of that couple
        marriage_distances += [int(D[indices[couple[0]], indices[couple[1]]])]

        # remove all possible pairings which included either of the now-married couple
        possible_couples = {pair:possible_couples[pair]
                                for pair in possible_couples
                                    if ((pair[0] != couple[0])
                                    and (pair[1] != couple[0])
                                    and (pair[0] != couple[1])
                                    and (pair[1] != couple[1]))}
        iter += 1
        # test1 = [k for k in possible_couples if k[0] == couple[0]]
        # test2 = [k for k in possible_couples if k[0] == couple[1]]
        # test3 = [k for k in possible_couples if k[1] == couple[0]]
        # test4 = [k for k in possible_couples if k[1] == couple[1]]
        # if test1 or test2 or test3 or test4:
        #     print("missed some possibilities")
        stay_single_forever = set([node[0] for node in possible_couples] + [node[1] for node in possible_couples])
    if iter == 0:
        # IE you never entered the while loop above
        stay_single_forever = {} 
    return unions, num_immigrants, marriage_distances, immigrants, wont_marry_until_next_time, len(stay_single_forever)


#%%
def add_children_edges_kolton(unions, num_people, child_probs):
    """
    PARAMETERS:
        unions: (list of tuple of int) marriages in the current generation
            (output of kolton_add_marriage_edges())
        num_people: (int) current total number of nodes (persons) in the graph
            (IE the sum of the size of every generation)
        child_probs: (dictionary) keys are number of children (int), values are
            the probability (float) that a couple has key children (the output
            of get_probabilities(child_dist))
        indices: (dictionary) mapping the current generations' names (int) to
            index (row/column number) in the current distance matrix D
    RETURNS:
        child_edges: (list of tuple of int) entries are (parent, child) and
            should be added to the graph
        families: (list of list of int) entries are lists of children pertaining
            to the ith couple (follows the order of unions)
        num_people + total_num_children: (int) updated total number of people in
            the community/graph after adding children to the current generation
            of marriages
        num_children_each_couple: (np.array of len(unions) of int) each entry is
            a random draw from the child_probs distribution, how many children
            the ith couple of unions has
        indices: (dictionary) mapping the current generations' names (int) to
            index (row/column number) in the current distance matrix D, updated
            to include the new children (new children are added to D outside of
            this function)
    """
    families = []
    child_edges = []

    num_children_each_couple = np.random.choice(np.array(list(child_probs.keys())), p=np.array(list(child_probs.values())), size=len(unions))
    total_num_children = sum(num_children_each_couple)
    # names = [num_people + child for child in range(total_num_children)]
    # families = [[names[k]] for k in range(last + family ) for family in num_children_per_couple if
    # families = [[num_people + child for child in range(family)] for family in num_children_per_couple]
    # families = [[child for child in range(] for family in num_children_per_couple]
    biggest_name = num_people

    for union, num_children in zip(unions, num_children_each_couple):
        if num_children == 0:
            families.append([])
        else:
            children = [biggest_name + 1 + child for child in range(num_children)]
            biggest_name += num_children  # the next 'name' to use, next available index
            father_edges = [(union[0], child) for child in children]
            mother_edges = [(union[1], child) for child in children]
            child_edges.append(father_edges + mother_edges)
            families.append(children)
    # flatten the list of family edges
    child_edges = [edge for family in child_edges for edge in family]
    #max_ind = max(indices.values())
    #indices = indices | {child + num_people: ind for ind, child in zip(range(max_ind+1, max_ind+1+total_num_children), range(total_num_children))}
    return child_edges, families, num_people + total_num_children, num_children_each_couple


# sum(get_probabilities(marriage_dist).values())
# sum(get_probabilities(child_dist, is_child=True).values())
# sum(get_probabilities(child_dist).values())
# set(child_dist)



#%%

# n = num_people
# people_retained = prev_generation_still_single
def update_distances_kolton(D, n, unions, families, indices, people_retained):
    """
    Build a distance matrix that keeps track of how far away each node is from
    each other. Need to update distances after new nodes added to graph (i.e.,
    after adding children)
    PARAMETERS:
        D (array): "old" matrix of distances, previous generation
            n (int): number of nodes currently in graph
        unions: (list of tuple of int) marriages in the current
            generation (output of kolton_add_marriage_edges())
        no_unions: (list of int) list of nodes in the current generation
            which did not marry
        families: (list of list of int) entries are lists of children
            pertaining to the ith couple (follows the order of unions)
            (output of add_children_edges_kolton())
        indices (dictionary): maps node name (an int) to index number
            (row/column number) in the current distance matrix D.
        people_retained (list): nodes which are unmarried, but which will attempt to
            marry someone in the next generation of people
    RETURNS:
        D1 (array): "new" (updated) matrix of distances
            for the current generation
        new_indices: (dictionary) mapping the current generations' names (int)
            to index (row/column number) in the current distance matrix D
    """
    # initialize new matrix
    num_children = len([child for fam in families for child in fam])
    num_people_retained = len(people_retained)

    D1 = np.zeros((num_children + num_people_retained,
                   num_children + num_people_retained))

    new_indices = {person:k for k, person in enumerate(people_retained + [child for fam in families for child in fam])}

    # check_indices(new_indices)
    if num_people_retained > 0:
        # the upper num_retained_people x num_retained_people block of D1 is just a slice from D
        # this builds out the "second quadrant" of the D1 matrix
        D1[new_indices[people_retained[0]]:new_indices[people_retained[-1]]+1,
           new_indices[people_retained[0]]:new_indices[people_retained[-1]]+1] = D[[indices[k] for k in people_retained]][:, [indices[k] for k in people_retained]]

        # now build out the distances from people_retained to the new generation (descendants of people_retained's cousins)
        # this builds out the "first and third" quadrants of the D1 matrix
        for rp, fam in product(people_retained, zip(unions, families)):
            # find minimum distance between the retained person and a representative child in the family
            father = fam[0][0]
            mother = fam[0][1]
            children = fam[1]  # all children in family will have same distance from retained person rp
            if len(children) == 0:
                # ie the union, family pair has no children listed,
                # the end of a line
                continue

            possible_distances = D[indices[rp], [indices[father], indices[mother]]]
            possible_distances = possible_distances[possible_distances > -1]
            d = np.min(possible_distances) + 1  # account for the additional edge between father/mother and ch
            D1[new_indices[rp], [new_indices[ch] for ch in children]] = d
            D1[[new_indices[ch] for ch in children], new_indices[rp]] = d


    # compute new distances, between new generation
    # this builds out the "fourth quadrant" of the D1 matrix
    unions = list(unions)
    for u, union in enumerate(unions):
        u_children = families[u]

        for other in unions[u:][1:]:
            o_children = families[unions.index(other)]

            # find all possible distances from union to other
            d1 = D[indices[union[0]]][indices[other[0]]]
            d2 = D[indices[union[1]]][indices[other[0]]]
            d3 = D[indices[union[0]]][indices[other[1]]]
            d4 = D[indices[union[1]]][indices[other[1]]]

            possible_distances = np.array([d1, d2, d3, d4])
            possible_distances = possible_distances[possible_distances > -1]  # IE where NOT infinite
            # compute distance between children of union and children of other
            d = np.min(possible_distances) + 2
            for uc in u_children:
                for oc in o_children:
                    D1[new_indices[uc]][new_indices[oc]] = d
                    D1[new_indices[oc]][new_indices[uc]] = d

        for c, ch in enumerate(u_children):
            for sibling in u_children[c:][1:]:
                D1[new_indices[ch]][new_indices[sibling]] = 2
                D1[new_indices[sibling]][new_indices[ch]] = 2

    return D1, new_indices


#%%
def check_indices(indices):
    all_ok = True
    min_child = min(indices.keys())
    max_child = max(indices.keys())
    for i in range(min_child, max_child-1,):
        if indices[i]+1 != indices[i+1]:
            all_ok = False
            print([(k, indices[k]) for k in range(i-2, i+2)])
    if all_ok:
        print("indices ok")


#%%
# TODO: what do we actually want this to return?
def human_family_network(num_people, marriage_dist, prob_finite_marriage, prob_inf_marriage, children_dist, name, when_to_stop=np.inf, num_gens=np.inf, save=True, out_dir='output'):
    """
    PARAMETERS:
        num_people (int): number of people (nodes) to include in initial
            generation
        marriage_dists: (list of int) one entry per marriage indicating how many
            generations between spouses (reported in the number of parent-child
            edges crossed so that distance between siblings is 2) in the
            specified dataset.  If no common ancestor (IE an infinite distance)
            then the corresponding entry is -1
        prob_finite_marriage (float): probability of marriage being drawn from
            the finite portion of marriage_dist (herein defined and treated as
            finite_marriage_probs)
        prob_inf_marriage (float): probability of marriage to a non-connected person
        children_dist: (list of int) one entry per pair of parents, indicating
            how many child edges each parent in the couple share
        name: (str) name for prefix of saved files
        when_to_stop: (int) target number of nodes to capture.  If supplied, the
            model will run until this target number of nodes (all together, not
            just the size of the current generation) is surpassed.
        num_gens (int): max number of generations for network to grow beyond the
            initial generation.  Default is np.inf.  If np.inf, then the model
            will run until the number of nodes in the example network is
            surpassed and then stop.

    RETURNS:
        G:
        D:
        unions:
        num_children_per_couple:
    """
    num_original_people = num_people
    total_num_single = 0 
    dies_out = False

    all_marriage_edges = []
    all_marriage_distances = []
    all_children_per_couple = []

    G = nx.DiGraph()
    # num_finite_dist = round(num_people * prob_finite_marriage)
    # num_inf_dist = round(num_people * prob_inf_marriage)
    # num_single = num_people - num_finite_dist - num_inf_dist
    marriage_dist_array = np.array(marriage_dist)
    finite_only_marriage_dist = marriage_dist_array[marriage_dist_array != -1]
    d = np.triu(np.random.choice(finite_only_marriage_dist, size=(num_people, num_people)), k=1)
    D = d + d.T
    indices = {node + 1:k for k, node in enumerate(range(num_people))}  # name:index
    generation_of_people = list(indices.keys())
    # explicitly add our first generation of nodes (otherwise we will fail to
    # add those who do not marry into our graph).  All future generations are
    # connected either through marriage or through parent-child arcs
    G.add_nodes_from(generation_of_people, layer=0)


    # build out an actual graph structure to back up the initial distances
    # imposed on generation 0
    for pair in combinations(generation_of_people, 2):

        nodes_to_add = [k for k in range(num_people+1, num_people+D[indices[pair[0]],
                                                                      indices[pair[1]]])]
        num_people += len(nodes_to_add)
        husband_line = nodes_to_add[:len(nodes_to_add)//2+1][::-1] + [pair[0]]   # path will go FROM first TO last node in list
        wife_line = nodes_to_add[len(nodes_to_add)//2:] + [pair[1]]  # path will go FROM first TO last node in list

        nx.add_path(G, husband_line, Relationship='Parent-Child')
        nx.add_path(G, wife_line, Relationship='Parent-Child')



    # get probabilities of possible finite distances to use in marriage function
    # and normalize it
    finite_marriage_probs = get_probabilities(marriage_dist)
    finite_marriage_probs = {key:value/sum(finite_marriage_probs.values()) for key, value in zip(finite_marriage_probs.keys(), finite_marriage_probs.values())}
    # now make the finite marriage entries sum to the probability of a finite marriage
    # for if we decide to add prob inf marriage, prob single to the dictionary 
    # finite_marriage_probs = {key:value*prob_finite_marriage for key, value in zip(finite_marriage_probs.keys(), finite_marriage_probs.values())}
    
    
    
    # now add an entry for infinite distance marriages
    # ??? TODO should we add the infinite distance portion the way rebbekah did?
    # marriage_probs[100] = (infdis/len(all_distances))/2  # include probability of infinite distance
    # factor = 1.0/sum(marriage_probs.values())   # normalizing factor
    # # normalize values for finite and infinite distances
    # for k in marriage_probs:
    #     marriage_probs[k] = marriage_probs[k]*factor
    # marriage_probs = finite_marriage_probs.copy()
    # marriage_probs[-1] = prob_inf_marriage
    # and add an entry for staying single
    # marriage_probs[0] = 1 - prob_finite_marriage - prob_inf_marriage

    # ditto for the child distribution
    child_probs = get_probabilities(children_dist, is_child=True)
    # ??? make probabilities non-negative (some entries are effectively zero, but negative)
    child_probs = {key:value if value > 0 else 0 for key, value in zip(child_probs.keys(), child_probs.values()) }
    child_probs = {key:value/sum(child_probs.values()) for key, value in zip(child_probs.keys(), child_probs.values())}

    # grow the network until there are the at least as many nodes (not counting
    # those created to impose the distances on generation 0, but counting those
    # in generation 0) as when_to_stop
    num_setup_people = num_people - num_original_people



    prev_generation_still_single = []
    current_prob_inf = np.inf 
    

    summary_statistics = []  # will hold ordered tuples of integers (# total people in graph,  # immigrants, # num_children, # num marriages, prob_inf_marriage, prob_finite_marriage, prob_inf_marriage(eligible_only), prob_finite_marriage(elegible_only))
    i = 1
    while (num_people - num_setup_people < when_to_stop) & (i < num_gens):
        # check_indices(indices)

        # for i in range(num_gens):
        # create unions between nodes to create next generation
        #unions, no_unions, all_unions, n, m, infdis, indices = add_marriage_edges(all_fam, all_unions, D, marriage_probs, p, ncp, n, infdis, indices)
        if len(generation_of_people) == 0:
            dies_out = True
            break
    
        # update your current finite marriage probabilities to favor those which are yet underrepresented
        if i > 1 and len(set(all_marriage_distances)) > 2:
            new_finite_marriage_probs = get_difference_in_probabilities(finite_marriage_probs, all_marriage_distances)
            new_prob_inf_marriage, new_prob_finite_marriage = get_difference_in_types_of_marriage(prob_inf_marriage, prob_finite_marriage, all_marriage_distances)
        # elif i > 1 and current_prob_inf > prob_inf_marriage:
        #     new_finite_marriage_probs = finite_marriage_probs 
        #     new_prob_inf_marriage, new_prob_finite_marriage = get_difference_in_types_of_marriage(prob_inf_marriage, prob_finite_marriage, all_marriage_distances)
        else:
            # if this is the first generation beyond the initial set up OR 
            # if you don't yet have more than one unique distance in your list 
            # of marriage edges, then just default to the unaltered finite-
            # distance marriage distribution 
            new_finite_marriage_probs = finite_marriage_probs
            new_prob_inf_marriage, new_prob_finite_marriage = prob_inf_marriage, prob_finite_marriage 
        print('gen. ', i)
        print('new_prob_inf:', new_prob_inf_marriage)
        print('new_prob_finite:', new_prob_finite_marriage)
        unions, num_immigrants, marriage_distances, immigrants, prev_generation_still_single, stay_single_forever = kolton_add_marriage_edges(generation_of_people, prev_generation_still_single, num_people, new_finite_marriage_probs, new_prob_inf_marriage, new_prob_finite_marriage, D, indices)
        total_num_single += stay_single_forever 
        # marriage edges should be undirected
        # ??? TODO should we add both directions in?  We will be able to grab just "Marriage" edges and then undirect this graph equivalently
        G.add_nodes_from(immigrants, layer=i-1)
        G.add_edges_from(unions, Relationship="Marriage")
        all_marriage_edges += list(unions)
        all_marriage_distances += marriage_distances

        for j in range(num_immigrants):
            # add non-connected people to distance matrix
            r = np.ones((1, len(indices) + 1 + j)) * -1  # -1 is infinite distance
            r[0, -1] = 0  # distance to self is 0
            c = np.ones((len(indices) + j, 1)) * -1  # -1 is infinite distance
            D = np.hstack((D, c))
            D = np.vstack((D, r))

        max_ind = max(indices.values())
        indices = indices | {immigrant + num_people + 1:ind for ind, immigrant in zip(range(max_ind + 1, max_ind+1+num_immigrants), range(num_immigrants))}  # +1 since we begin counting people at 1, 2, ... not at 0, 1, ...
        stats = [num_people - num_setup_people, num_immigrants]
        num_people += num_immigrants

        # add children to each marriage
        child_edges, families, num_people, num_children_per_couple = add_children_edges_kolton(unions, num_people, child_probs)

        # update distances between nodes
        D, indices = update_distances_kolton(D, num_people, unions, families, indices, prev_generation_still_single)

        generation_of_people = [key for key in indices.keys() if key not in prev_generation_still_single]  # only grab the new people 
        G.add_nodes_from(generation_of_people, layer=i)
        G.add_edges_from(child_edges, Relationship='Parent-Child')
        all_children_per_couple += list(num_children_per_couple)

        # stats.append(len(generation_of_people))
        stats.append(sum(num_children_per_couple))
        stats.append(len(unions))
        stat_prob_marry = len(all_marriage_distances) * 2 / len(G) 
        stat_frac_inf = sum(np.array(all_marriage_distances) == -1) / len(all_marriage_distances)
        stats.append(stat_prob_marry * stat_frac_inf)  
        stats.append(stat_prob_marry * (1 - stat_frac_inf)) 
        
        # now recalculate marriage stats using only the eligible nodes 
        # (IE not those preceding gen 0, not prev_gen_still_single, and not 
        # leaf nodes---only those nodes that were given the chance to marry)
        stat_prob_marry = len(all_marriage_distances) * 2 / (len(G) - num_setup_people - len(prev_generation_still_single) - sum(num_children_per_couple))
        current_prob_inf = stat_prob_marry * stat_frac_inf
        stats.append(stat_prob_marry * stat_frac_inf) 
        stats.append(stat_prob_marry * (1 - stat_frac_inf)) 
        
        summary_statistics.append(stats)
        
        i += 1

        # ??? save output at each generation
    if save:
        output_path = makeOutputDirectory(out_dir, name)
        df = pd.DataFrame(data=summary_statistics, columns=['num_people (excluding initial setup)', 'num_immigrants', 'num_children', 'num_marriages', 'prob_inf_marriage', 'prob_finite_marriage', 'prob_inf_marriage(eligible_only)', 'prob_finite_marriage(eligible_only)'])
        df.index.name='generation'
        df.to_csv(os.path.join(output_path, str(name)+'_summary_statistics.csv'))
        Gname = Gname = "{}/{}_G.gpickle".format(output_path, name)   # save graph
        nx.write_gpickle(G, Gname)
        Uname = "{}/{}_marriage_edges".format(output_path, name) + '.pkl'   # save unions
        with open(Uname, 'wb') as fup:
            pickle.dump(all_marriage_edges, fup)
        Dname = "{}/{}_marriage_distances".format(output_path, name) +'.pkl' # save marriage distances
        with open(Dname, 'wb') as myfile:
            pickle.dump(all_marriage_distances, myfile)
        Cname = "{}/{}_children_per_couple".format(output_path, name) + '.pkl'  # save children
        with open(Cname, 'wb') as fcp:
            pickle.dump(all_children_per_couple, fcp)

        paj = format_as_pajek(G, name)
        with open('{}/model-{}-oregraph.paj'.format(output_path, name), 'w') as o:
            o.writelines(paj)

        # # save output of the last generation
        # if i == gen:
        #     print("Last generation: ", i+1)
        #     Gname = "{}_G{}.gpickle".format(name, i+1)   # save graph
        #     nx.write_gpickle(G, Gname)
        #     Dname = "{}_D{}.npy".format(name, i+1)   # save D
        #     np.save(Dname, D)
        #     indicesname = "{}_indices{}.npy".format(name, i)  # save indices
        #     np.save(indicesname, indices)
        #     Uname = "{}_U{}".format(name, i+1)   # save unions
        #     with open(Uname, 'wb') as fup:
        #         pickle.dump(all_unions, fup)
        #     Cname = "{}_C{}".format(name, i+1)   # save children
        #     with open(Cname, 'wb') as fcp:
        #         pickle.dump(all_children, fcp)

        return G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, output_path

    else:
        return G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out

#%%
"""
below is example code to run the model
"""
# #
# name = 'obidos'
# num_people = 5
# name = 'arara'
# num_people = 7
# # name = 'san_marino'
# # num_people = 100
# name = 'trio_1960s'
# num_people = 3

# marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)


# children_dist = child_dist
# when_to_stop = size_goal 
# num_gens = np.inf
# G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, output_path = human_family_network(num_people, marriage_dist, prob_finite_marriage, prob_inf_marriage, child_dist, name, save=True, when_to_stop=size_goal)

#%%
def find_start_size(name, out_directory='start_size', filename='start_size', max_iters=100, dies_out_threshold=5,  verbose=False, save_start_sizes=True, random_start=True): # n = number of initial nodes
    counter = 0

    filename = name + '_' + filename
    marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)
    greatest_lower_bound = 2
    least_upper_bound = size_goal

    if random_start:
        num_people = np.random.randint(greatest_lower_bound, size_goal)
    else:
        num_people = size_goal//2
    dies_out = 0 # counter for the number of times the model dies out

    start_sizes = [num_people]
    while dies_out != dies_out_threshold: # while the number of times the model dies out is not equal to the threshold of dying:

        for i in range(max_iters):
            counter += 1
            G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies = human_family_network(num_people,
                                                                                                                marriage_dist,
                                                                                                                prob_finite_marriage,
                                                                                                                prob_inf_marriage,
                                                                                                                child_dist,
                                                                                                                name,
                                                                                                                when_to_stop=size_goal,
                                                                                                                save=False)
            if dies:
                dies_out += 1
            if dies_out > dies_out_threshold:
                break

        if greatest_lower_bound >= least_upper_bound - 1:
            # IE the ideal lies between these two integers
            # so return the larger
            num_people = least_upper_bound
            break
        elif dies_out == dies_out_threshold:
            break
        elif dies_out > dies_out_threshold:  # we want to increase num_people
            greatest_lower_bound = num_people  # current iteration died out too frequently.  Won't need to search below this point again.
            num_people = (num_people + least_upper_bound) // 2 # midpoint between num_people and size_goal
            dies_out = 0

        elif dies_out < dies_out_threshold: # we want to decrease num_people
            least_upper_bound = num_people  # current iteration died out too infrequently.  Won't need to search above this point again
            num_people = (greatest_lower_bound + num_people) // 2 # midpoint between 2 and num_people
            dies_out = 0

        if verbose:
            print('greatest_lower_bound: ', greatest_lower_bound)
            print('least_upper_bound: ', least_upper_bound)
            print('starting population: ', num_people)
        start_sizes.append(num_people)


    if save_start_sizes:
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        filename = find_file_version_number(out_directory, filename, extension='.txt')
        # save a text file (one integer per line)
        with open(os.path.join(out_directory, filename +'.txt'), 'w') as outfile:
            outfile.writelines([str(k) + '\n' for k in start_sizes])
        # save the actual object
        with open(os.path.join(out_directory, filename + '.pkl'), 'wb') as outfile:
            pickle.dump(start_sizes, outfile)

    return start_sizes, counter
#%%

def find_start_size2(name,
                     out_directory='start_size',
                     filename='start_size',
                     max_iters=100,
                     dies_out_threshold=5,
                     verbose=False,
                     save_start_sizes=True,
                     random_start=True,
                     load_data=False,
                     marriage_dist=None,
                     num_marriages=None,
                     prob_inf_marriage=None,
                     prob_finite_marriage=None,
                     child_dist=None,
                     size_goal=None):
    filename = name + '_' + filename
    if load_data:
        marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)

    greatest_lower_bound = 2
    least_upper_bound = size_goal

    if random_start:
        num_people = np.random.randint(greatest_lower_bound, size_goal)
    else:
        num_people = size_goal//2
    dies_out = 0 # counter for the number of times the model dies out

    start_sizes = [num_people]

    max_iters = max_iters - dies_out_threshold
    while dies_out != dies_out_threshold: # while the number of times the model dies out is not equal to the threshold of dying:

        for i in range(max_iters):
            G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies = human_family_network(num_people,
                                                                                                                marriage_dist,
                                                                                                                prob_finite_marriage,
                                                                                                                prob_inf_marriage,
                                                                                                                child_dist,
                                                                                                                name,
                                                                                                                when_to_stop=size_goal,
                                                                                                                save=False)
            if dies:
                dies_out += 1
            if dies_out > dies_out_threshold:
                break


        if greatest_lower_bound >= least_upper_bound - 1:
            # IE the ideal lies between these two integers
            # so return the larger
            num_people = least_upper_bound
            break
        elif dies_out == dies_out_threshold:
            break
        elif dies_out > dies_out_threshold:  # we want to increase num_people
            greatest_lower_bound = num_people  # current iteration died out too frequently.  Won't need to search below this point again.
            num_people = (num_people + least_upper_bound) // 2 # midpoint between num_people and size_goal
            dies_out = 0

        elif dies_out < dies_out_threshold: # we want to decrease num_people
            least_upper_bound = num_people  # current iteration died out too infrequently.  Won't need to search above this point again
            num_people = (greatest_lower_bound + num_people) // 2 # midpoint between 2 and num_people
            dies_out = 0

        if verbose:
            print('greatest_lower_bound: ', greatest_lower_bound)
            print('least_upper_bound: ', least_upper_bound)
            print('starting population: ', num_people)
        start_sizes.append(num_people)


    if save_start_sizes:
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        filename = find_file_version_number(out_directory, filename, extension='.txt')
        # save a text file (one integer per line)
        with open(os.path.join(out_directory, filename +'.txt'), 'w') as outfile:
            outfile.writelines([str(k) + '\n' for k in start_sizes])
        # save the actual object
        with open(os.path.join(out_directory, filename + '.pkl'), 'wb') as outfile:
            pickle.dump(start_sizes, outfile)

    return start_sizes
#%%

def repeatedly_call_start_size(name, out_directory='start_size', iters=5, max_iters=100, dies_out_threshold=5,  verbose=False, save_start_sizes=True, save_individual_start_sizes=False, random_start=True, show_plot=False):
    #find out directory.  Every iteration in this function call will output to the same file
    out_dir = makeOutputDirectory(out_directory, name)

    if save_start_sizes:
        # create the folder, text file to which EACH start_size list will be appended
        filename = name + '_' + 'start_size'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(os.path.join(out_dir, filename + '.txt')):
            with open(os.path.join(out_dir, filename + '.txt'), 'w'):
                pass

    to_plot = []
    for i in range(iters):
        start_sizes = find_start_size(name,
                                      out_dir,
                                      max_iters=max_iters,
                                      dies_out_threshold=dies_out_threshold,
                                      verbose=verbose,
                                      save_start_sizes=save_individual_start_sizes,
                                      random_start=random_start)
        to_plot.append(start_sizes)

        if save_start_sizes:
            # save a text file (one line per run of find_start_size())
            with open(os.path.join(out_dir, filename +'.txt'), 'a') as outfile:
                outfile.write(str(start_sizes))
                outfile.write('\n')

    if save_start_sizes:
        # save the unaltered (entries will be of differing lengths) set of start sizes
        # as a pickle object for later use
        with open(os.path.join(out_dir, filename + '.pkl'), 'wb') as outfile:
            pickle.dump(to_plot, outfile)

    # prep each entry of to_plot.  Not every iteration will have the same num
    # of entries.  Just repeat the last entry as necessary
    length = np.max([len(k) for k in to_plot])
    for start_sizes in to_plot:
        while len(start_sizes) < length:
            start_sizes.append(start_sizes[-1])

    fig = plt.figure(figsize=(12,9), dpi=300)
    for k in range(len(to_plot) -1):
        plt.plot(to_plot[k], color='k', linewidth=0.5, alpha=0.65)

    # plot the last one with a label
    plt.plot(to_plot[-1], color='k', linewidth=0.5, alpha=0.65, label='individual run')
    # make the plot display in text the final average starting value
    avg_run = np.mean(to_plot, axis=0)
    plt.text(length - 1.25, avg_run[-1]+3, str(round(avg_run[-1])), fontsize=16)
    # now plot the average, bolded
    plt.plot(avg_run, color='k', linewidth=7, alpha=0.8, label='mean')
    plt.xticks([k for k in range(length)], labels=[k for k in range(1, length+1)], fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(name + ' initial population search', fontsize=24)
    plt.legend()
    plt.ylabel('starting population', fontsize=16)
    plt.xlabel('iterations', fontsize=16)
    plt.savefig(os.path.join(out_dir, name + '_starting_size_graph.png'), format='png')
    if show_plot:
        plt.show()

    return avg_run
#%%

# temp = find_start_size(name, max_iters=5, dies_out_threshold=1)

#%%