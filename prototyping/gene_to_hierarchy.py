import pandas as pd
import numpy as np

def find_number_of_levels(data):
    max_levels = 0
    for index, row in data.iterrows():
        levels = 1
        parent = row['parent_node_id']
        leaf = row['node_id']
        trace = []
        trace.append(leaf)
        while parent not in [np.nan, 'AMR', 'ALL']:
            trace.append(parent)
            parent = parent.replace("'", "\'")
            levels += 1
            parent = data[data['node_id'] == parent]['parent_node_id'].item()
        if levels > max_levels:
            max_levels = levels
            print("max levels is now {} after processing {} with trace {}".format(max_levels, leaf, trace))
    return max_levels




data = pd.read_csv('/media/david/T7/NCBI/ReferenceGeneHierarchy.txt', sep='\t')
ml = find_number_of_levels(data)
print("Max levels is {}".format(ml))