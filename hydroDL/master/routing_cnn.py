"""这里是用keras的cnn框架计算河道汇流的各个函数。"""
from datetime import datetime
import os
import torch.utils.data as Data  # 用于创建 DataLoader
import time
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from explore import trans_norm
from hydroDL.model import crit
from hydroDL.model.cnn import TryCnn
from pytorchtools import EarlyStopping
from utils.hydro_util import send_email
import seaborn as sns
import math
sns=sns.set(style="whitegrid",color_codes=True)


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from skorch.callbacks import Callback


class AccuracyGmail(Callback):
    # TODO
    def __init__(self, min_accuracy):
        self.min_accuracy = min_accuracy

    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        if self.critical_epoch_ > -1:
            return
        # look at the validation accuracy of the last epoch
        if net.history[-1, 'valid_acc'] >= self.min_accuracy:
            self.critical_epoch_ = len(net.history)

    def on_train_end(self, net, **kwargs):
        if self.critical_epoch_ < 0:
            msg = "Accuracy never reached {} :(".format(self.min_accuracy)
        else:
            msg = "Accuracy reached {} at epoch {}!!!".format(
                self.min_accuracy, self.critical_epoch_)

        send_email("training end", msg)


def to_supervised(data, n_input):
    """convert history into inputs and outputs
    将以日为行，以年为列的各个站点的数据转换为以站点为行，日期为列，列数暂定14，的矩阵
     Parameters
    ----------
    df : 几个series时间序列组成的list.
    n_input : 输入的计算的长度，时间维度上的长度

    Returns
    -------
    inputs : ndarray
        输入矩阵的形式是二维的，每行代表一天，每列代表一个站点，从第一天开始，n_input天组成的一个array做输入
    outputs: ndarray
        第15天yicahng站数据做输出
    """
    # 拼接完成之后，将矩阵拆解为多组输入输出
    # 最大天数设为14
    inputs = []
    outputs = []
    for j in range(data.shape[0] - n_input - 1):
        input = data[j:j + n_input]
        # 输出取最后一列的 number, that is number of final site-- yichang
        output = data[j + n_input][-1]
        inputs.append(input)
        outputs.append(output)
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    inputs= np.delete(inputs,5,axis = 2)
    return inputs, outputs


# 接下来搭建cnn计算框架，用keras现成的框架直接计算，train the model
# def build_model(train_data, valid_data, batch_size, num_epochs, criterion, optimizer):
#     """
#     :param train_data: 训练集 dataloader
#     :param batch_size: 训练数据分块个数
#     :param num_epochs: 训练代数
#     :return: 回归分析后的CNN模型实例化对象. and  A `History` object-- Its `History.history` attribute is
#         a record of training loss values and metrics values
#         at successive epochs, as well as validation loss values
#         and validation metrics values (if applicable).
#     """
#     net = try_cnn()
#     if torch.cuda.is_available():
#         net = net.cuda()
#     for epoch in range(num_epochs):
#         prev_time = datetime.now()
#         train_loss = 0
#         train_acc = 0
#         net = net.train()
#         for x_train, y_train in train_data:
#             if torch.cuda.is_available():
#                 x_train = x_train.cuda()  # (bs, 3, h, w)
#                 y_train = y_train.cuda()  # (bs, h, w)
#             # forward
#             output = net(x_train)
#             loss = criterion(output, y_train)
#             # backward
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#
#         cur_time = datetime.now()
#         h, remainder = divmod((cur_time - prev_time).seconds, 3600)
#         m, s = divmod(remainder, 60)
#         time_str = "Time %02d:%02d:%02d" % (h, m, s)
#         if valid_data is not None:
#             valid_loss = 0
#             valid_acc = 0
#             net = net.eval()
#             for x_train, y_train in valid_data:
#                 if torch.cuda.is_available():
#                     x_train = x_train.cuda()
#                     y_train = y_train.cuda()
#                 output = net(x_train)
#                 loss = criterion(output, y_train)
#                 valid_loss += loss.item()
#             epoch_str = (
#                     "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
#                     % (epoch, train_loss / len(train_data),
#                        train_acc / len(train_data), valid_loss / len(valid_data),
#                        valid_acc / len(valid_data)))
#         else:
#             epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
#                          (epoch, train_loss / len(train_data),
#                           train_acc / len(train_data)))
#         prev_time = cur_time
#         print(epoch_str + time_str)
#     return net


def evaluate_forecasts(observed, predicted):
    """
    评价预报结果，几种常用方式：rmse，r-squared，nse，qualified rate
    :param observed: ndarray
        测试数据对应的观测值
    :param predicted:
        测试数据模型的输出
    :return:dict--my_rmse, r_squared, my_nse, qualified_rate
    """
    # use the evaluator with the Root Mean Square Error (objective function 3)
    my_rmse = np.sqrt(mean_squared_error(observed, predicted))
    # my_rmse = evaluator(rmse, predicted, observed)
    # 用sklearn计算r-squared
    r_squared = r2_score(observed, predicted)
    # 计算NSE 一般用于验证水文模型模拟结果的好坏
    # my_nse = evaluator(nse, predicted, observed)
    # 计算合格率，根据《水文预报》第四版328页中描述，过程预报许可误差规定为预见期内实测变幅的20%作为许可误差，
    # 当该流量小于实测值的5%时，以该值为许可误差。这个标准貌似有点太严格了，按照Lu Chen论文及其引用的文献中标书
    # 合格的定义是模拟与实测差值在实测值正负20%内。和论文保持一致
    diffs = abs(observed - predicted)
    # 按照书上定义计算
    qualified_book_num = 0
    # 按论文中定义计算
    qualified_num = 0
    for i in range(1, observed.size):
        variation = abs(observed[i] - observed[i - 1])
        if variation * 0.2 < observed[i] * 0.05:
            qualified_error = observed[i] * 0.05
        else:
            qualified_error = variation * 0.2
        if diffs[i] <= qualified_error:
            qualified_book_num = qualified_book_num + 1
    for j in range(observed.size):
        if diffs[j] <= (observed[j] * 0.2):
            qualified_num = qualified_num + 1

    qualified_rate_book = qualified_book_num / (observed.size - 1)
    qualified_rate = qualified_num / observed.size
    return {"RMSE": my_rmse, "R2": r_squared, "QR": qualified_rate}


def cnn_train(try_cnn:TryCnn,
              input,
              output,
              crit,
              patience,
              epochs:int,
              seq_first:bool=True):
    """train a model using data from dataloader"""
    print("Start Training...")
    # input = input.reshape(-1, 1, 84)
    # x = input.reshape(-1, 1, 1, input.shape[2])
    x = input.reshape(-1, 1, input.shape[1], input.shape[2])
    y = output.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    print(x_train, x_test, y_train, y_test)
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    # 建立训练数据的 DataLoader
    training_dataset = Data.TensorDataset(x_train, y_train)
    # 把dataset放到DataLoader中
    train_data_loader = Data.DataLoader(dataset=training_dataset, batch_size=95, shuffle=True)
    # 建立验证数据的 DataLoader
    valid_dataset = Data.TensorDataset(x_test, y_test)
    # 把dataset放到DataLoader中
    valid_data_loader = Data.DataLoader(dataset=valid_dataset, batch_size=95, shuffle=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    if torch.cuda.is_available():
        crit = crit.cuda()
        try_cnn = try_cnn.cuda()
    for epochs in range(1, epochs + 1):  # loop over the dataset multiple times
        ###################
        # train the model #
        ###################
        try_cnn.train()  # prep model for training
        t0 = time.time()
        for batch_idx, (batch_xs, batch_ys) in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            optimizer = torch.optim.Adam(try_cnn.parameters(),0.001)
            optimizer.zero_grad()
            # if seq_first:
            #     batch_xs = batch_xs.transpose(0, 1)
            #     batch_ys = batch_ys.transpose(0, 1)
            if torch.cuda.is_available():
                batch_xs = batch_xs.cuda()
                batch_ys = batch_ys.cuda()
            # forward + backward + optimize
            outputs = try_cnn(batch_xs)
            # criterion = torch.nn.MSELoss(reduction='mean')
            criterion = crit.RmseLoss1d()
            loss = criterion(outputs, batch_ys)
            loss.backward()
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            # print('Epoch: ', epoch, '| Step: ', step, '| loss_avg: ', running_loss / steps_num)
        ######################
        # validate the model #
        try_cnn.eval()  # prep model for evaluation
        for data, target in valid_data_loader:
            # if seq_first:
            #     data = data.transpose(0, 1)
            #     target = target.transpose(0, 1)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = try_cnn(data)
            # calculate the loss
            criterion = crit.RmseLoss1d()
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        val_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(val_loss)
        epoch_len = len(str(epochs))

        print_msg = (f'[{epochs:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {val_loss:.5f}')
        log_str = 'time {:.2f}'.format(time.time() - t0)
        print(print_msg, log_str)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        early_stopping(val_loss, try_cnn)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    print('Finished Training')
    return x_train,try_cnn, avg_train_losses, avg_valid_losses

def cnn_test(try_cnn:TryCnn,X,Y,stat_dict):
    try_cnn.load_state_dict(torch.load("F:\科研类\codes\hydro-routing-cnn\checkpoint.pt"))
    # x = X.reshape(-1, 1, X.shape[1], X.shape[2])
    x = X.reshape(-1, 1, X.shape[1], X.shape[2])
    # x1= X.reshape(715, 1,-1)
    # x = x1.reshape(-1, 1, 1, x1.shape[2])
    y = Y.reshape(-1, 1)
    x = torch.from_numpy(x).float()
    try_cnn.eval()
    batch_size=100
    batches=math.ceil(x.shape[0]/batch_size)
    preds=[]
    for i in range(batches):
        test_x=x[i*batch_size:(i+1)*batch_size]
        pred_x = try_cnn(test_x)
        preds.extend(pred_x)
    pred=torch.Tensor(preds)
    pred = pred.detach().numpy()
    print( "pred求平均：",pred.mean())
    # pred_value = trans_norm(pred.reshape(1, pred.shape[0], pred.shape[1]), 'flow', stat_dict, to_norm=False)
    pred_value = trans_norm(pred.reshape(1, pred.shape[0], 1), 'flow', stat_dict, to_norm=False)
    obs_value = trans_norm(y.reshape(1, y.shape[0], 1), 'flow', stat_dict, to_norm=False)
    return obs_value.flatten(), pred_value.flatten()


# def cnn_train(input, output, batch_size, epochs, model_dict,stat_dict):
#     """train the model whose params come from bo result"""
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # train_ds = RoutingDataset(X, Y)
#     x = input.reshape(-1, 1, input.shape[1], input.shape[2])
#     y = output.reshape(-1, 1)
#     x_train = torch.from_numpy(x).float()
#     y_train = torch.from_numpy(y).float()
#     net = NeuralNet(
#
#             try_cnn,
#             max_epochs=epochs,
#             batch_size=batch_size,
#             criterion=crit.RmseLoss1d,
#             optimizer=torch.optim.Adam,
#             lr=1e-3,
#             callbacks=[EarlyStopping(patience=20)],
#             device=device)
# net.fit(x, y)
# x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1,random_state=1)
# print(x_train, x_test, y_train, y_test )
# x_train = torch.from_numpy(x_train).float()
# x_test = torch.from_numpy(x_test).float()
# y_train = torch.from_numpy(y_train).float()
# y_test = torch.from_numpy(y_test).float()
# valid_ds = dataset.Dataset(x_test, y_test)
# print(valid_ds)
#
# net = NeuralNet(
#     try_cnn,
#     max_epochs=epochs,
#     batch_size=batch_size,
#     criterion=crit.RmseLoss1d,
#     optimizer=torch.optim.Adam,
#     lr=1e-3,
#     callbacks=[EarlyStopping(patience=20)],
#     device=device,
#     train_split=predefined_split(valid_ds)
# )
# # net.fit(train_ds)

#  net.fit(x_train, y_train)
#  pred_train=net.predict(x_train)
#  print(pred_train)
#  pred_train_value=trans_norm(pred_train.reshape(1, pred_train.shape[0], pred_train.shape[1]), 'flow',stat_dict ,to_norm=False)
#  obs_train_value = trans_norm(y_train.reshape(1, y.shape[0], 1), 'flow', stat_dict,to_norm=False)
#  obs_train_value=obs_train_value.flatten()
#  pred_train_value=pred_train_value.flatten()
#  print(obs_train_value,pred_train_value)
#
#  my_rmse = np.sqrt(mean_squared_error(obs_train_value, pred_train_value))
#  r_squared = r2_score(obs_train_value, pred_train_value)
# #计算qr
#  diffs = abs(obs_train_value - pred_train_value)
#  qualified_book_num = 0
#  qualified_num = 0
#  for i in range(1, obs_train_value.size):
#      variation = abs(obs_train_value[i] - obs_train_value[i - 1])
#      if variation * 0.2 < obs_train_value[i] * 0.05:
#          qualified_error = obs_train_value[i] * 0.05
#      else:
#          qualified_error = variation * 0.2
#      if diffs[i] <= qualified_error:
#          qualified_book_num = qualified_book_num + 1
#  for j in range(obs_train_value.size):
#      if diffs[j] <= (obs_train_value[j] * 0.2):
#          qualified_num = qualified_num + 1
#  qualified_rate_book = qualified_book_num / (obs_train_value.size - 1)
#  qualified_rate = qualified_num /obs_train_value.size
#  print("RMSE: ",my_rmse, "R2: ",r_squared,  "QR: ",qualified_rate)
#
#
#  out_f_params = os.path.join(model_dict["dir"]["Out"], 'some-file.pkl')
#  out_f_optimizer = os.path.join(model_dict["dir"]["Out"], 'opt.pkl')
#  out_f_history = os.path.join(model_dict["dir"]["Out"], 'history.json')
#  net.save_params(f_params=out_f_params, f_optimizer=out_f_optimizer, f_history=out_f_history)
#  return net


# def cnn_train_forecast(input, output,stat_dict, history=None, model_file=None, weight_file=None):
#
#     if history is None:
#         loaded_model = NeuralNet(
#             try_cnn,
#             criterion=torch.nn.MSELoss,
#         )
#         loaded_model.initialize()  # This is important!
#         loaded_model.load_params(f_params="F:\\科研类\\codes\\hydro-routing-cnn\\example\\output\\sanxia\\some-file.pkl")
#     else:
#         loaded_model = history.model
#         X = input.reshape(-1, 1, input.shape[1], input.shape[2])
#         y = output.reshape(-1, 1)
#         x = torch.from_numpy(X).float()
#         x_train_pred = loaded_model.predict(x)
#         cnn_pred_value = trans_norm(x_train_pred.reshape(1, x_train_pred.shape[0], x_train_pred.shape[1]), 'flow', stat_dict, to_norm=False)
#         cnn_obs_value = trans_norm(y.reshape(1, y.shape[0], 1), 'flow', stat_dict, to_norm=False)
#      return cnn_pred_value.flatten(),  cnn_obs_value.flatten()
#







# def cnn_test(X, Y, stat_dict, history=None, model_file=None, weight_file=None):
#     """test the model whose params come from bo result"""
#     if history is None:
#         loaded_model = NeuralNet(
#             try_cnn,
#             criterion=torch.nn.MSELoss,
#         )
#         loaded_model.initialize()  # This is important!
#         loaded_model.load_params(f_params="F:\\科研类\\codes\\hydro-routing-cnn\\example\\output\\sanxia\\some-file.pkl")
#     else:
#         loaded_model = history.model
#     x = X.reshape(-1, 1, X.shape[1], X.shape[2])
#     y = Y.reshape(-1, 1)
#     x = torch.from_numpy(x).float()
#     pred = loaded_model.predict(x)
#     pred_value = trans_norm(pred.reshape(1, pred.shape[0], pred.shape[1]), 'flow', stat_dict, to_norm=False)
#     obs_value = trans_norm(y.reshape(1, y.shape[0], 1), 'flow', stat_dict, to_norm=False)
#     # pred = loaded_model.predict(X)
#     # pred_value = trans_norm(pred.reshape(1, pred.shape[0], pred.shape[1]), 'flow', stat_dict, to_norm=False)
#     # obs_value = trans_norm(Y.reshape(1, Y.size, 1), 'flow', stat_dict, to_norm=False)
#     return obs_value.flatten(), pred_value.flatten()
