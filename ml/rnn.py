import sys, json, argparse, random, re, os, shutil
sys.path.append('src/')
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
# import nni
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
# from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch_geometric.data import Data, Batch, DataLoader
import networkx as nx
from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from src.model.basic_model import *
from utils.data_convert import generate_samples
from src.trafficDataset import TrafficDataset
import pdb
from src.model import continue_learning
from torch_geometric.utils import to_dense_batch 

# import torch.optim.ReduceLROnPlateau as ReduceLROnPlateau

pin_memory = True
n_work = 3

def update(src, tmp):
    for key in tmp:
        src[key] = tmp[key]


def init(args):    
    conf_path = os.path.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    path = os.path.join(info['model_path'], info['logname'] + info["time"])
    ct.mkdirs(path)
    info['log_dir'] = path
    update(vars(args), info)
    del info


def init_log(args):
    log_dir, log_filename = args.log_dir, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, log_filename+'.log'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('logger name:%s', os.path.join(log_dir, log_filename+'.log'))
    vars(args)['logger'] = logger
    return logger


def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def train(inputs, args):
    # model setting
    path = os.path.join(args.model_path, args.logname+args.time, str(args.year))
    ct.mkdirs(path)
    # writer = SummaryWriter(os.path.join(path, "tsborad"))
    model = Basic_Model(args).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if args.scheduler == 'cos': scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.001)
    elif args.scheduler == 'epo': scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
    if args.loss == 'mse': lossfunc = func.mse_loss
    elif args.loss == 'huber': lossfunc = func.smooth_l1_loss
    
    total_time = 0.0
    #### dataset definition
    train_loader = DataLoader(TrafficDataset(inputs, 'train'), batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=n_work)
    val_loader = DataLoader(TrafficDataset(inputs, 'val'), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
    test_loader = DataLoader(TrafficDataset(inputs, 'test'), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
    args.logger.info("[*] Dataset load!")


    args.logger.info("[*] start")
    global_train_steps = len(train_loader) // args.batch_size +1
   
    iters = len(train_loader)
    lowest_validation_loss = 1e7
    counter = 0
    patience = 10
    for epoch in range(args.epoch):
        training_loss = 0.0
        start_time = datetime.now()
        # train model
        cn = 0
        for batch_idx, data in enumerate(train_loader):
            # data_time = datetime.now()
            if args.scheduler == 'cos':
                scheduler.step(epoch + batch_idx/iters)
            data = data.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            pred = model(data)
            if args.strategy == 'incremental_only' and args.year!=args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.node_list, :]
                data.y = data.y[:, args.node_list, :]
            loss = lossfunc(data.y, pred, reduction='mean')
            training_loss += float(loss)
            loss.backward()
            optimizer.step()
            
            # loss_time = datetime.now() - loss_time
            # print("loss time:",loss_time.total_seconds())
            total_time += (datetime.now() - start_time).total_seconds()
            cn += 1
            # if cn == math.ceil(len(train)/args.batch_size): 
            #     writer.add_scalar('training_loss', training_loss/cn, epoch * len(train) + batch_idx)
        training_loss = training_loss/cn 
 
        # validate model
        validation_loss = 0.0
        # validation_loss = test(model, val, args.device, args.n, pin_memory)
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device,non_blocking=pin_memory)
                pred = model(data)
                if args.strategy == 'incremental_only' and args.year!=args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.node_list, :]
                    data.y = data.y[:, args.node_list, :]
                loss = lossfunc(data.y, pred)
                validation_loss += loss
                cn += 1
                # if cn == math.ceil(len(val)/args.batch_size): 
                #     writer.add_scalar('validation_loss', validation_loss/cn, epoch * len(val) + batch_idx)
        validation_loss = float(validation_loss/cn)
        
        # scheduler.step(validation_loss)
        if args.scheduler == 'epo':
            scheduler.step()
        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        if args.nni:
            nni.report_intermediate_result(validation_loss)

        # early stopping 
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save(model, os.path.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break

    best_model_path = os.path.join(path, str(lowest_validation_loss)+".pkl")
    best_model = torch.load(best_model_path, args.device)
    # test model
    test_model(model, args, test_loader, pin_memory)
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred = model(data)
            loss += func.mse_loss(data.y, pred, reduction='mean')
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss/cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        # print(np.concatenate(pred_, 0).shape)
        pred_ = np.concatenate(pred_, 0).reshape((-1,args.graph_size,12))
        truth_ = np.concatenate(truth_, 0).reshape((-1,args.graph_size,12))
        # print("pred shape ", pred_.shape)
        # print(pred_.shape, truth_.shape)
        mae = metric(truth_, pred_, args.graph_size, args.logger)
        if args.nni == 1:
            nni.report_final_result(mae)
        return loss


def metric(ground_truth, prediction, node, logger):
    pred_time = [3,6,12]
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
    return mae


def main(args):
    # seed_set()
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):
        vars(args)['year'] = year
        inputs = generate_samples(osp.join(args.save_data_path, str(year)), np.load(osp.join(args.raw_data_path, str(year)+'.npz'))['x'], graph) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+'.npz'))

        train(inputs, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--conf', type = str, default = 'conf/test.json')
    parser.add_argument('--nni', type = int, default = 0, help='0: non nni, 1: nni applied, 2: fixed some parameters directly from the tuning result of nni')
    parser.add_argument('--paral', type = int, default = 0)
    parser.add_argument('--logname', type = str, default = 'info')
    args = parser.parse_args()
    init(args)

    if args.nni == 1:
        nni_params = nni.get_next_parameter()
        update(vars(args), nni_params)
    # 根据需要重写nni_params
    elif args.nni == 2:
        nni_params = {"lr": 0.006, "oc": 32, "oc_t": 16, "oc_gcn": 32, "k": 6, "n_gcn": 3, "loss": "mse", "bs":128}
        update(vars(args), nni_params)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != '-1' else "cpu"
    vars(args)['device'] = device
    main(args)