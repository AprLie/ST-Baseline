from thundersvm import SVR
import argparse
from utils import *
import time
import numpy as np 
import os, sys
import os.path as osp
import logging
import tqdm
parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument("--path", type = str, default = "res")
parser.add_argument("--gpuid", type = int, default = 2)
parser.add_argument("--begin_year", type = int, default = 2011)
parser.add_argument("--end_year", type = int, default = 2017)
parser.add_argument("--logname", type = str, default = "info")
parser.add_argument("--save_data_path", type = str, default = "/home/v-xuche3/project/dyna_traffic/data_process/F11T17/FastData", help="0: training first year, 1: load from model path of first year")
# parser.add_argument("--first_year_model_path", type = str, default = "res/district3F11T17/upperbound2021-01-19-15:38:06.565639/2011/16.4423.pkl")
#res/district3F11T17/upperbound2021-01-18-09:12:15.523789/2011/4127.2555.pkl", res/district3F11T17/upperbound2021-01-17-12:59:34.540254/2011/4146.1628.pkl", help="model path of first year, load when load_first_year=1")
args = parser.parse_args()

def mkdirs(path):
    if not osp.exists(path):
        os.makedirs(path)


def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
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



logger = init_log(args)
result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}, "total_time":{}, "average_time":{}}
for i in range(12):
    locals()["clf"+str(i)] = SVR(max_iter=100, cache_size=10240, gpu_id=1)

for i in range(2011, 2018):
    datas = 1
    if datas:
        data = np.load(osp.join(args.save_data_path, str(i)+"_30day.npz"))
        # T, len, N
        train_x, train_y, val_x, val_y, test_x, test_y = data["train_x"], data["train_y"], data["val_x"], data["val_y"], data["test_x"], data["test_y"]
    else:
        train_x = np.random.rand(100,12,120) 
        train_y = np.random.rand(100,12,120) 
        test_x = np.random.rand(100,12,120)  
        test_y = np.random.rand(100,12,120) 
    print(train_x.shape, train_y.shape)
    graph_size = train_x.shape[2]
    samples = test_x.shape[0]
    pred = np.zeros((samples, 12, graph_size))
    start_time = time.time()
    # print(pred.shape)

    for j in range(12):
        for k in tqdm.tqdm(range(graph_size)):
            locals()["clf"+str(j)].fit(train_x[:,:,k], train_y[:,j,k])
            y = locals()["clf"+str(j)].predict(test_x[:,:,k])
            # print(y.shape)
            pred[:,j,k] = y
    use_time = time.time()-start_time
    logger.info("year {} time {:.4f}".format(i, use_time))
    result["total_time"][i] = use_time
    # print(time.time()-start_time)
    # 
    for j in [3,6,12]:
        mae = masked_mae_np(test_y[:,:j,:], pred[:,:j,:],0)
        mape = masked_mape_np(test_y[:,:j,:], pred[:,:j,:],0)
        rmse = masked_mse_np(test_y[:,:j,:], pred[:,:j,:],0) ** 0.5
        result[j]["mae"][i] = mae
        result[j]["mape"][i] = mape
        result[j]["rmse"][i] = rmse
        logger.info("{}\tmae\t{:.2f}\tmape\t{:.2f}\trmse\t{:.2f}".format(j,mae, mape, rmse))

for i in [3, 6, 12]:
    for j in ['mae', 'rmse', 'mape']:
        info = ""
        for year in range(args.begin_year, args.end_year+1):
            if i in result:
                if j in result[i]:
                    if year in result[i][j]:
                        info+="{:.2f}\t".format(result[i][j][year])
        logger.info("{}\t{}\t".format(i,j) + info)

for year in range(args.begin_year, args.end_year+1):
    if year in result:
        info = "year\t{}\ttotal_time\t{}".format(year, result[year]["total_time"])
        logger.info(info)

        
    
    # print(y.shape)
