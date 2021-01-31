"""Data that will be inputted to model"""
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from explore import *


class DataModel(object):
    """data formatter， utilizing function of DataSource object to read data and transform"""

    def __init__(self, data_source, *args):
        """:parameter data_source: DataSource object"""
        self.data_source = data_source
        # call "read_xxx" functions of DataSource to read forcing，flow，attributes data
        if len(args) == 0:
            # read flow
            data_flow = data_source.read_streamflow()
            data_flow, usgs_id, t_range_list = data_source.screen_streamflow(data_flow)
            self.data_flow = data_flow
            # wrap gauges and time range to a dict.
            # To guarantee the time range is a left-closed and right-open interval, t_range_list[-1] + 1 day
            self.t_s_dict = OrderedDict(sites_id=usgs_id,
                                        t_final_range=[np.datetime_as_string(t_range_list[0], unit='D'),
                                                       np.datetime_as_string(
                                                           t_range_list[-1] + np.timedelta64(1, 'D'), unit='D')])
            # statistics
            stat_dict = self.cal_stat_all()
            self.stat_dict = stat_dict

        else:
            self.data_flow = args[0]
            self.t_s_dict = args[1]
            self.stat_dict = args[2]

    def cal_stat_all(self):
        """calculate statistics of streamflow, forcing and attributes. 计算统计值，便于后面归一化处理。"""
        # streamflow
        flow = self.data_flow
        stat_dict = dict()
        stat_dict['flow'] = cal_stat(flow)
        return stat_dict

    def get_data_obs(self, rm_nan=True, to_norm=True):
        """径流数据读取及归一化处理，会处理成三维，最后一维长度为1，表示径流变量"""
        stat_dict = self.stat_dict
        data = self.data_flow
        # 为了调用trans_norm函数进行归一化，这里先把径流变为三维数据
        data = np.expand_dims(data, axis=2)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        data = trans_norm(data, 'flow', stat_dict, to_norm=to_norm)
        return data

    def load_data(self, model_dict):
        """读取数据为模型输入的形式，完成归一化运算
        :parameter
            model_dict: 载入数据需要模型相关参数
        :return  np.array
            x: 3-d  gages_num*time_num*var_num
            y: 3-d  gages_num*time_num*1
            c: 2-d  gages_num*var_num
        """
        # 如果读取到统计数据的json文件，则不需要再次计算。
        opt_data = model_dict["data"]
        # TODO:there are some nan value in streamflow, transform to 0 temporally
        rm_nan_y = opt_data['rmNan'][0]
        y = self.get_data_obs(rm_nan=rm_nan_y)
        y = y.reshape(y.shape[0], y.shape[1]).T
        return y


class RoutingDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.X = x.reshape(-1, 1, x.shape[1], x.shape[2])
        self.Y = y.reshape(-1, 1)

    def __getitem__(self, idx):
        x_i = torch.from_numpy(self.X[idx]).float()
        y_i = torch.from_numpy(self.Y[idx]).float()
        return x_i, y_i

    def __len__(self):
        return self.X.shape[0]
