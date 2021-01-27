import numpy as np 
import pandas as pd 
import os 
import os.path as osp




def graph_convert(load_path, save_path, days=-1):
    # type: graph: directly save data: save to [T, N]

    graph = np.load(load_path)["x"]
    if days != -1:
        graph = graph[0:days*288, :]
        print(graph.shape)
    np.savetxt(save_path, graph, delimiter=",")


graph_path = "../district3F11T17/STGCN_graph"
data_path = "/home/v-xuche3/project/dyna_traffic/data_process/F11T17/finaldata"

for year in range(2011, 2018):
    # graph_convert(osp.join(graph_path, str(year)+"_adj.npz"), osp.join(graph_path, str(year)+"_adj.csv"))
    graph_convert(osp.join(data_path, str(year)+".npz"), osp.join(".", str(year)+"_data.csv"), days=31)