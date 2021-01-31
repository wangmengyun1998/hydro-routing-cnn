"""利用贝叶斯优化进行超参数优选。用交叉验证和贝叶斯优化对机器学习模型CNN调参"""
import os
import torch
from bayes_opt import BayesianOptimization, JSONLogger
from bayes_opt.event import Events
from skorch import NeuralNet
from skorch.callbacks import EarlyStopping

from data.data_model import RoutingDataset
from hydroDL.master.routing_cnn import to_supervised, cnn_train
from hydroDL.model import crit
from hydroDL.model.cnn import TryCnn


def cnn_cv(X, y, batch_size, epochs, model_dict):
    """CNN cross validation.
    使用scikit-learn和keras进行交叉验证.
    因为用于贝叶斯优化，评价目标是一个单一的指标，所以就直接用回归的评价指标即可
    """
    net = cnn_train(X, y, batch_size, epochs, model_dict)
    valid_loss_final = net.history[-1, 'batches', -1, 'valid_loss']
    return -valid_loss_final


def optimize_cnn(data_model):
    """ Apply Bayesian Optimization to CNN parameters"""
    model_dict = data_model.data_source.data_config.model_dict

    def cnn_crossval(n_input, batch_size, epochs):
        """Wrapper of CNN cross validation.
        主要是处理下读取输入输出，离散化等问题
        Parameters
        ----------
        n_input : 输入选择的时间长度.
        batch_size: define parameters,batch是指把整体数据分成若干份，因为一次性输入所有数据计算比较困难
        epochs:指模型的多次训练，以使之收敛，数据越多样，epoch越大
        """
        # 首先要对数据进行离散化处理，否则后面会报错
        n_input = 16  # int(n_input)
        batch_size = int(batch_size)
        epochs = int(epochs)
        # 调用to_supervised将数据分为输入和输出部分
        data, targets = to_supervised(data_model.load_data(model_dict), n_input)
        return cnn_cv(X=data, y=targets, batch_size=batch_size, epochs=epochs, model_dict=model_dict)

    optimizer = BayesianOptimization(
        f=cnn_crossval,
        pbounds={"n_input": model_dict["train"]["inputRange"], "batch_size": model_dict["train"]["batchRange"],
                 "epochs": model_dict["train"]["epochRange"]},  # interval
        random_state=1234,
        verbose=2
    )
    bo_logger = JSONLogger(path=os.path.join(model_dict["dir"]["Out"], "bo_logs.json"))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, bo_logger)
    optimizer.maximize(init_points=2, n_iter=2)
    print("Final result:", optimizer.max)
