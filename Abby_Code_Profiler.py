from family_network_model import *
from matplotlib import pyplot as plt
import networkx as nx
import ast
import pickle
from get_model_parameters import *
import ast
from visualize_ph_lines_abby_mod import *
from vis_outside_line import *
from persim import bottleneck

if __name__ == "__main__":
    pass

def run_model():

    graphs, names = get_graphs_and_names(sort=True)

    def run_model(name,n,gen):

        with open('./UnionDistances/other_parameters.txt') as f:
            params = f.readline()
        data_parameters = ast.literal_eval(params)
        # save the following parameters:
        m_e = data_parameters[name][0]  # number of marriage edges
        P = data_parameters[name][1]    # probability of marriage
        NCP = data_parameters[name][2]  # probability of nonconnected marriage

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



        # initialize all parameters
        name = name + '_test1' # CHANGE

        marriage_dist = nx_distances
        children_dist = nx_children
        p = P
        ncp = NCP
        infdis = round((inf_dis/m_e - (NCP/2))*m_e)


        # run model
        G, D, unions, children, infdis = human_family_network(n, gen, marriage_dist, p, ncp, infdis, children_dist, name)
        return G,D,unions,children,infdis

    #CHANGE ONLY THIS CELL
    name = 'tory' #put the name of the network you want to study
    iters = 5 #how many to test for statistical significance
    n_ls = [9] #put all the different values for n that you want to test
    gen_ls = [9] #put all the different values for gen that you want to test

    def test_gen_and_n(name,n,gen,iters=5,dim=1):

        for item in names:
            if name in item:
                i = names.index(item)

        real_G = graphs[i]
        zero_real,one_real,two_real = Ripser(real_G)
        if dim == 0:
            one_real = zero_real
        if dim == 1:
            pass
        if dim == 2:
            one_real = two_real

        norm_dim1 = []
        bottleneck_dim1 = []
        num_of_nodes = []

        one_real = list(one_real)
        one_real.sort(key = lambda x: x[1])
        one_real.sort(key = lambda x: x[0])
        ylist_real = [i for i in range(len(one_real))]
        xlist_real = []
        for i in range(len(one_real)):
            xlist_real.append(one_real[i][1])
        bottleneck_dim1.append(bottleneck(one_real,one_real))
        real_num_of_nodes = real_G.number_of_nodes()

        for i in range(iters):
            approx_G,D,unions,children,infdis = run_model(name,n,gen)
            num_of_nodes.append(approx_G.number_of_nodes())
            zero_approx,one_approx,two_approx = Ripser(approx_G)
            one_approx = list(one_approx)
            one_approx.sort(key = lambda x: x[1])
            one_approx.sort(key = lambda x: x[0])
            ylist_approx = [i for i in range(len(one_approx))]
            xlist_approx = []
            for i in range(len(one_approx)):
                xlist_approx.append(one_approx[i][1])

            bottleneck_dim1.append(bottleneck(one_approx,one_real))
            norm_dim1.append(np.abs(np.linalg.norm(xlist_approx)-np.linalg.norm(xlist_real)))

        print(f'Average bottleneck distance: {sum(bottleneck_dim1)/len(bottleneck_dim1)}.')
        print(f'Average norm: {sum(norm_dim1)/len(norm_dim1)}.')
        return sum(bottleneck_dim1)/len(bottleneck_dim1),sum(norm_dim1)/len(norm_dim1),sum(num_of_nodes)/len(num_of_nodes),real_num_of_nodes

    bottleneck_avgs = []
    norm_avgs = []
    num_of_nodes_avgs = []

    for g,nn in zip(gen_ls,n_ls):
        bottleneck_avg, norm_avg, num_of_nodes_avg, real_num_of_nodes = test_gen_and_n(name,nn,g,iters=iters)
        bottleneck_avgs.append(bottleneck_avg)
        norm_avgs.append(norm_avg)
        num_of_nodes_avgs.append(num_of_nodes_avg)