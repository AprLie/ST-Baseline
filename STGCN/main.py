# -*- coding:utf-8 -*-

import os
import argparse
import sys
import mxnet as mx
from mxnet import nd
import logging, time
from utils import math_graph
from data_loader import data_utils
from model.trainer import model_train
import os.path as osp
ctx = mx.gpu(2)

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_vertices', type=int, default=228)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--order_of_cheb', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--adj_path', type=str,
                    default='datasets/district3F11T17/STGCN_graph')
parser.add_argument('--time_series_path', type=str,
                    default='datasets/district3F11T17/STGCN_data')
parser.add_argument('--path', type=str,
                    default='res/')
parser.add_argument('--logname', type=str,
                    default='trafficstream')

args = parser.parse_args()
vars(args)["result"] = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}, "total_time":{}, "mean_time":{}}

def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+str(time.time())+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger
    return logger

print('Training configs: {}'.format(args))
init_log(args)
n_his, n_pred = args.n_his, args.n_pred
order_of_cheb = args.order_of_cheb

# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]
log_dir = 'log'
for year in range(2011,2018):
    vars(args)["year"] = year
    
    adj = math_graph.weight_matrix(os.path.join(args.adj_path, str(year)+'_adj.csv'))
    vars(args)["num_of_vertices"] = adj.shape[0]
    args.logger.info("year {} graph size {}".format(year, args.num_of_vertices))
    L = math_graph.scaled_laplacian(adj)
    cheb_polys = nd.array(math_graph.cheb_poly_approx(L, order_of_cheb))

    # Data Preprocessing
    
    PeMS_dataset = data_utils.data_gen(os.path.join(args.time_series_path, str(year)+'_data.csv'), n_his + n_pred)
    args.logger.info('>> Loading dataset with Mean: {0:.2f}, STD: {1:.2f}'.format(
        PeMS_dataset.mean,
        PeMS_dataset.std
    ))
    cur_log_dir = os.path.join(log_dir, str(year))
    if not os.path.exists(cur_log_dir): os.makedirs(cur_log_dir)
    # if __name__ == '__main__':
    #     import shutil
    #     logdir = './logdir'
    #     if os.path.exists(logdir):
    #         shutil.rmtree(logdir)
    model_train(blocks, args, PeMS_dataset, cheb_polys, ctx, logdir=cur_log_dir)
result = args.result
for i in [3, 6, 12]:
    for j in ['mae', 'rmse', 'mape']:
        info = ""
        for year in range(2011, 2018):
            if i in result:
                if j in result[i]:
                    if year in result[i][j]:
                        info+="{:.2f}\t".format(result[i][j][year])
        args.logger.info("{}\t{}\t".format(i,j) + info)

for year in range(2011, 2018):
    if year in result:
        info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(year, result[year]["total_time"], result[year]["average_time"], result[year]['epoch_num'])
        args.logger.info(info)

