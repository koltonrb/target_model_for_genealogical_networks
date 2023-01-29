# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 11:48:02 2023

@author: kolto
"""

import pandas as pd 
import numpy as np
import os 
from get_model_parameters import get_names
from family_model import makeOutputDirectory
import ast
import pickle 

def find_files(out_directory, filename, ):
    ver = 1
    output_dir = os.path.join(out_directory, filename + '_')
    file_list = []
    while os.path.exists(output_dir+str(ver)):
        if os.path.exists(output_dir+str(ver)):
            file_list.append(output_dir+str(ver))
        ver += 1
    
    return file_list 

#%%

def evaluate_start_size_searches(start_size_dir='./start_size', save_report_to='data_set_stats'):
    report = pd.DataFrame(columns=['name', 'ran_to_completion', 'iters', 'avg_start_size'])
    
    names = get_names(save_paths=False)
    names.remove('warao')

    completed_networks = set()
    partially_completed_networks = set()
    no_info_networks = set()
    for name in names: 
        # attempt to load the pkl file (repeatedly_call_start_size only saves a pkl file if the run went to completion (all iterations specified, without an error))
        file_list = find_files(start_size_dir, name)
        for folder in file_list:
            try:
                start_size_lists = pd.read_pickle(os.path.join(folder, name+'_start_size.pkl'))
                completed_networks.add(name)
                avg_start_size = int(np.round(np.mean([row[-1] for row in start_size_lists])))
                report = pd.concat([report, pd.Series({'name':name, 'ran_to_completion':True, 'iters': 50, 'avg_start_size':avg_start_size}).to_frame().T], ignore_index=True)
            except: 
                
                with open(os.path.join(folder, name+'_start_size.txt')) as f:
                    start_size_lists = [ast.literal_eval(row.strip()) for row in f.readlines()]
                num_iters = len(start_size_lists)
                if num_iters > 0:
                    partially_completed_networks.add(name)
                    avg_start_size = avg_start_size = int(np.round(np.mean([row[-1] for row in start_size_lists])))
                    report = pd.concat([report, pd.Series({'name':name, 'ran_to_completion':False, 'iters':num_iters, 'avg_start_size':avg_start_size}).to_frame().T], ignore_index=True)
                else:
                    no_info_networks.add(name)
                    report = pd.concat([report, pd.Series({'name':name, 'ran_to_completion':False, 'iters':num_iters, 'avg_start_size':np.nan}).to_frame().T], ignore_index=True)
                    
        
    report = report.drop_duplicates()
    
    out_dir = makeOutputDirectory(save_report_to, 'review_of_start_sizes')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with pd.ExcelWriter(os.path.join(out_dir, 'data_set_start_sizes.xlsx'), engine='xlsxwriter') as writer:
        report.to_excel(writer)
    report.to_pickle(os.path.join(out_dir, 'data_set_start_sizes.pkl'))
    
    with open(os.path.join(out_dir, 'successful_networks.txt'), 'w') as o:
       for n in sorted(list(completed_networks)):
           o.write(n)
           o.write('\n')
    with open(os.path.join(out_dir, 'successful_networks.pkl'), 'wb') as o:
        pickle.dump(sorted(list(completed_networks)), o)
    
    with open(os.path.join(out_dir, 'partially_completed_networks.txt'), 'w') as o:
       # those that ran for some but not all specified iterations 
        for n in sorted(list(partially_completed_networks)):
           o.write(n)
           o.write('\n')
    with open(os.path.join(out_dir, 'partially_completed_networks.pkl'), 'wb') as o:
        pickle.dump(sorted(list(partially_completed_networks)), o)
    
    with open(os.path.join(out_dir, 'did_not_run_networks.txt'), 'w') as o:
        # text file was created, but never written to
        for n in sorted(list(no_info_networks)):
            o.write(n)
            o.write('\n')
    with open(os.path.join(out_dir, 'did_not_run_networks.pkl'), 'wb') as o:
        pickle.dump(sorted(list(no_info_networks)), o)

