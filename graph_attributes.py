import networkx as nx
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import shlex
import os
from networkx.algorithms.moral import moral_graph

def print_functions():
    print('content_list: returns list of rows from pajek file, used in other functions')
    print('separate_parts: separates sections from pajek file, used in other functions')
    print('draw_graph: draws graph with colored edges (red = marriage, blue = parent-child)')
    print('\tand shaped nodes (M = triangle, F = circle)')
    print('graph_with_attributes: creates a nx graph object with attributes')
    print('count_attributes: returns a list of the number of nodes or edges for each attribute')
    print('\tmale, female, unknown, marriage edges, parent-child edges')
    print('get_graphs_and_names: creates a list of graphs with attributes from Original_Sources,')
    print('\tand returns a corresponding list of names')
    print('get_attrs_array: returns a np array with all of the atttribue counts.')
    print('\trows are individuals, columns follow count_attributes order')

def content_list(graph_name):
    # open and read graph file
    with open(graph_name, 'r') as file:
        contents = file.readlines()

    # clean up graph file
    for i in range(len(contents)):
        contents[i] = contents[i].rstrip() # remove trailing whitespace
    contents = np.array(contents)
    noempties = [item != '' for item in contents] # get rid of elements with empty strings
    return list(contents[noempties]) # turn it back into a list and return

def separate_parts(graph_name,edge_type):
    """Count the number of edges in a pajek graph file.
           Parameters:
               graph_name (str):
               edge_type (str):
           Returns:
               nodes (list of lists): rows are people, columns are attributes
               marriages (list): list of tuples representing marriage edges
               pc (list): list of tuples representing parent child edges
    """
    # read in and format contents of pajek file
    contents = content_list(graph_name)
    # get the correct edge-type
    if edge_type == 'M':
        start_str = '*edges' # represent marriages
        alter_str = '*arcs'
    elif edge_type == 'PC':
        start_str = '*arcs'  # represent parent-child relationships
        alter_str = '*edges'
    elif edge_type == 'V':
        start_ind = 2
    elif edge_type == 'A':
        return separate_parts(graph_name,'V'), separate_parts(graph_name, 'M'), separate_parts(graph_name,'PC')
    # return contents list (for verification purposes)
    if edge_type == 'c':
        return contents
    def get_tuples(e_list):
        # turn edge_lists into list of tuples
        for i in range(len(e_list)):
            t,f,l = e_list[i].split()
            e_list[i] = (int(t),int(f))
        return e_list
    # get vertex list and attribute dictionary
    if edge_type == 'V':
        start_ind = 2
        try:
            e_ind = contents.index('*edges')
        except:
            e_ind = 0
        try:
            a_ind = contents.index('*arcs')
        except:
            e_ind = 0
        alter_ind = min(a_ind,e_ind)
        # get names and genders
        name_dict = {}
        gender_dict = {}
        for node in contents[start_ind:alter_ind]:
            try:
                num,name,gen = shlex.split(node)
                num = int(num)
            except:
                print('Vertex Error after' + str(num))
            if gen == 'triangle':
                gen = 'M'
            elif gen == 'ellipse' or 'circle':
                gen = 'F'
            elif gen == 'diamond' or 'square':
                gen = 'U'
            else:
                print('Error in gender:', node)
            name_dict.update({num:name}) # add name to dict
            gender_dict.update({num:gen}) # add gender to dict

        return [alter_ind-start_ind,name_dict,gender_dict]
    # count edges
    else:
        try:
            start_ind = contents.index(start_str)
        except:
            return 0
        try:
            alter_ind = contents.index(alter_str)
        except:
            alter_ind = 0
        if start_ind < alter_ind:
            return get_tuples(contents[start_ind+1:alter_ind])
        else:
            return get_tuples(contents[start_ind+1:])

def draw_paj_graph(graph_name,k = 1):
    """Takes in a PAJEK formatted graph file and draws the graph in an easy to interpret form.
    Marriage edges are red. Parent child edges are blue. Males are triangles, Females are circles."""
    # get graph data
    nodes,marr_edges,pc_edges = separate_parts(graph_name,'A')
    # create graph and add nodes
    g = nx.Graph()
    g.add_nodes_from(np.arange(nodes[0])+1)
    g.add_edges_from(marr_edges)
    g.add_edges_from(pc_edges)
    # get genders
    male = list(np.array(list(nodes[2].keys()))[np.array(list(nodes[2].values()))=='M'])
    female = list(np.array(list(nodes[2].keys()))[np.array(list(nodes[2].values()))=='F'])
    unknown = list(np.array(list(nodes[2].keys()))[np.array(list(nodes[2].values()))=='U'])
    # get position of nodes and edges
    pos = nx.kamada_kawai_layout(g)
    # draw graph
    nx.draw_networkx_edges(g,pos,marr_edges,edge_color='r')
    nx.draw_networkx_edges(g,pos,pc_edges,edge_color='b')
    nx.draw_networkx_nodes(g,pos,male,node_shape = 'v',node_color='k',node_size=15)
    nx.draw_networkx_nodes(g,pos,female,node_shape='o',node_color='k',node_size=10)
    nx.draw_networkx_nodes(g,pos,unknown,node_shape='x',node_color='g',node_size=15)
    plt.title(graph_name[30:-13])
    plt.show()
    
def draw_nx_graph(graph,k = 1,pc_only = False, pos = None):
    """Takes in a nx graph object and draws the graph in an easy to interpret form.
    Marriage edges are red. Parent child edges are blue. Males are triangles, Females are circles."""
    # get k-core
    g = nx.k_core(graph,k)
    # get gender attributes
    nodes = np.array(list(nx.get_node_attributes(g,'Gender').keys()))
    gen = np.array(list(nx.get_node_attributes(g,'Gender').values()))
    m_mask = np.where(gen == 'M')
    f_mask = np.where(gen == 'F')
    u_mask = np.where(gen == 'U')
    male = list(nodes[m_mask])
    female = list(nodes[f_mask])
    unknown = list(nodes[u_mask])
    # get edge attributes
    nodes = np.array(list(nx.get_edge_attributes(g,'Relationship').keys()))
    rel = np.array(list(nx.get_edge_attributes(g,'Relationship').values()))
    marr_mask = rel == 'Marriage'
    pc_mask = rel == 'Parent-Child'
    marr_edges = list(nodes[marr_mask])
    pc_edges = list(nodes[pc_mask])
    # get position of nodes and edges
    if pos == None:
        try:
            pos = nx.kamada_kawai_layout(g)
        except:
            pos = nx.spring_layout(g)
    # draw graph
    if pc_only == False:
        nx.draw_networkx_edges(g,pos,marr_edges,edge_color='r')
    nx.draw_networkx_edges(g,pos,pc_edges,edge_color='b')
    nx.draw_networkx_nodes(g,pos,male,node_shape = 'v',node_color='k',node_size=15)
    nx.draw_networkx_nodes(g,pos,female,node_shape='o',node_color='k',node_size=10)
    nx.draw_networkx_nodes(g,pos,unknown,node_shape='x',node_color='g',node_size=15)
    plt.show()


def graph_with_attributes(graph_name, directed = False,pc_only = False):
    """ Input lists, return a nx.Graph with correct node and edge attributes.
    """
    # get graph data
    nodes,marr_edges,pc_edges = separate_parts(graph_name,'A')
    # create graph and add nodes
    if directed == False:
        g = nx.Graph()
        g.add_nodes_from(np.arange(nodes[0])+1)
        if pc_only == False:
            try:
                g.add_edges_from(marr_edges)
            except:
                print('Error: marriage edges from',graph_name)
        try:
            g.add_edges_from(pc_edges)
        except:
                print('Error: parent-child edges from',graph_name)
    else:
        g = nx.DiGraph()
        g.add_nodes_from(np.arange(nodes[0])+1)        
        g.add_edges_from(pc_edges)    
    # assign names and gender attributes
    nx.set_node_attributes(g,nodes[1],'Name')
    nx.set_node_attributes(g,nodes[2],'Gender')
    # assign edge types: marriage vs parent child
    edge_type_dict = {}
    if directed == False:
        for edge in marr_edges:
            edge_type_dict.update({edge:'Marriage'})
    for edge in pc_edges:
        edge_type_dict.update({edge:'Parent-Child'})
    nx.set_edge_attributes(g,edge_type_dict,'Relationship')
    # return graph with attributes
    return g

def count_attributes(graph):
    """
    Input:
        nx graph
    Output:
        Males
        Females
        Unknown
        Marriage edges
        Parent child edges
    """
    # get gender node counts
    vals = list(nx.get_node_attributes(graph,'Gender').values())
    male = vals.count('M')
    female = vals.count('F')
    unknown = vals.count('U')
    # get relationship edge counts
    vals = list(nx.get_edge_attributes(graph,'Relationship').values())
    marrs = vals.count('Marriage')
    pc = vals.count('Parent-Child')
    #return counts
    return male, female, unknown ,marrs, pc

def get_graphs_and_names(path = './Original_Sources/',directed = False,sort = False):
    """Create list of nx graphs from original sources"""
    name_list = os.listdir(path)
    n = len(name_list)
    for i in range(n):
        name_list[i] = path+name_list[i]
    # get graphs and names
    graph_list = []
    graph_names = []
    for g in name_list:
        try:
            graph_list.append(graph_with_attributes(g,directed))
            graph_names.append(g)
        except:
            print('Graph failed:',g)
    if sort == True:
        nodes = np.array([g.number_of_nodes() for g in graph_list])
        order = np.argsort(nodes)
        g_list = [graph_list[n] for n in order]
        g_names = [graph_names[n] for n in order]
        return g_list,g_names
    else:
        return graph_list, graph_names

def get_attrs_array(graph_list):
    """input graph list (from get_graphs_and_names) to get an array of attributes
    rows: individual graphs
    columns: male, female, unknown, marriage edges, parent-child edges
    """
    atr_list = []
    for i in range(len(graph_list)):
        m,f,u,mar,pc = count_attributes(graph_list[i])
        atr_list.append([m,f,u,mar,pc])
    return np.array(atr_list)

def walk_out(G,node,distance,verbose = False):
    nodes = set()
    nodes.update({node})
    for i in range(distance):
        m = list(nodes)
        if verbose == True:
            print(m)
        for n in m:
            nodes.update(set(G[n]))
    return G.subgraph(nodes)
