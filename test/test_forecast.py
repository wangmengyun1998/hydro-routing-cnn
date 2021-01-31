import os
import unittest

from bayes_opt.util import Colours

import definitions
from data import *
from hydroDL.master import routing_cnn, cnn_test, evaluate_forecasts
from utils import *
from hydroDL import *
from visual import *


class MyTestCase(unittest.TestCase):
    config_file = definitions.CONFIG_FILE
    project_dir = definitions.ROOT_DIR
    dataset = 'sanxia'
    dir_db = os.path.join(project_dir, 'example/data', dataset)
    dir_out = os.path.join(project_dir, 'example/output', dataset)
    dir_temp = os.path.join(project_dir, 'example/temp', dataset)
    data_source_dump = os.path.join(dir_temp, 'data_source_test.txt')
    stat_file = os.path.join(dir_temp, 'Statistics_test.json')
    flow_file = os.path.join(dir_temp, 'flow_test.npy')
    t_s_dict_file = os.path.join(dir_temp, 'dictTimeSpace_test.json')

    obs_file = os.path.join(dir_temp, 'observe')
    pred_file = os.path.join(dir_temp, 'predict')
    obs_file_npy = os.path.join(dir_temp, 'observe.npy')
    pred_file_npy = os.path.join(dir_temp, 'predict.npy')

    def test_forecast_data_model(self):
        print(Colours.yellow("--- reading CNN---"))
        config_file = definitions.CONFIG_FILE
        # 读取模型配置文件
        config_data = DataConfig(config_file)
        # 准备训练数据
        source_data = DataSource(config_data, config_data.model_dict["data"]["tRangeTest"])
        # 构建输入数据类对象
        model_data = DataModel(source_data)
        # 序列化保存对象
        dir_temp = source_data.all_configs["temp_dir"]
        source_file = self.data_source_dump
        stat_file = self.stat_file
        flow_file = os.path.join(dir_temp, 'flow_test')
        t_s_dict_file = self.t_s_dict_file

        # 存储data_model，因为data_model里的数据如果直接序列化会比较慢，所以各部分分别序列化，dict的直接序列化为json文件，数据的HDF5
        serialize_pickle(source_data, source_file)
        serialize_json(model_data.stat_dict, stat_file)
        serialize_numpy(model_data.data_flow, flow_file)
        serialize_json(model_data.t_s_dict, t_s_dict_file)

    def test_forecast(self):
        # 进行模型训练
        # train model
        print(Colours.yellow("--- CNN forecast---"))
        source_data = unserialize_pickle(self.data_source_dump)
        stat_dict = unserialize_json(self.stat_file)
        data_flow = unserialize_numpy(self.flow_file)
        t_s_dict = unserialize_json(self.t_s_dict_file)
        model_data = DataModel(source_data, data_flow, t_s_dict, stat_dict)
        model_dict = model_data.data_source.data_config.model_dict
        model_file = os.path.join(model_dict["dir"]["Out"], "model.yaml")
        weight_file = os.path.join(model_dict["dir"]["Out"], "weights.h5")

        bo_json_file = os.path.join(model_dict["dir"]["Out"], "bo_logs.json")
        bo_json = unserialize_bo_json(bo_json_file)
        # n_input = int(bo_json["params"]["n_input"])
        n_input = int(16)
        batch_size = int(100)
        epochs = int(100)
        data, targets = routing_cnn.to_supervised(model_data.load_data(model_dict), n_input)
        obs_value, pred_value = cnn_test(data, targets, stat_dict, model_file=model_file, weight_file=weight_file)
        print("the observe value:", obs_value)
        print("the predict value:", pred_value)
        serialize_numpy(obs_value, self.obs_file)
        serialize_numpy(pred_value, self.pred_file)

    def test_plot(self):
        observe_flow = unserialize_numpy(self.obs_file_npy)
        predict_flow = unserialize_numpy(self.pred_file_npy)
        inds_dict = evaluate_forecasts(observe_flow, predict_flow)
        print(inds_dict)
        plot_box_inds(subset_of_dict(inds_dict, ["RMSE"]))
        plot_box_inds(subset_of_dict(inds_dict, ["QR", "R2"]))
        #plot_box_inds(subset_of_dict(inds_dict, ["NSE", "QR", "R2"]))


if __name__ == '__main__':
    unittest.main()

