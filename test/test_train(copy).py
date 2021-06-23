import os
import unittest

import definitions
from data import *
from hydroDL.master import to_supervised, cnn_train
from refine import *
from utils import *


class MyTestCase(unittest.TestCase):
    config_file = definitions.CONFIG_FILE
    project_dir = definitions.ROOT_DIR
    dataset = 'sanxia'
    dir_db = os.path.join(project_dir, r'example\data', dataset)
    dir_out = os.path.join(project_dir, r'example\output', dataset)
    dir_temp = os.path.join(project_dir, r'example\temp', dataset)
    data_source_dump = os.path.join(dir_temp, 'data_source.txt')
    stat_file = os.path.join(dir_temp, 'Statistics.json')
    flow_file = os.path.join(dir_temp, 'flow.npy')
    t_s_dict_file = os.path.join(dir_temp, 'dictTimeSpace.json')
 #类中每一个测试用例运行前都会运行，用例的初始化
    def setUp(self):
        print('setUp...读取datamodel')
        # 存储data_model，因为data_model里的数据如果直接序列化会比较慢，所以各部分分别序列化，dict的直接序列化为json文件，数据的HDF5
        self.source_data = unserialize_pickle(self.data_source_dump)
        self.stat_dict = unserialize_json(self.stat_file)
        self.data_flow = unserialize_numpy(self.flow_file)
        self.t_s_dict = unserialize_json(self.t_s_dict_file)
        self.data_model = DataModel(self.source_data, self.data_flow, self.t_s_dict, self.stat_dict)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(self.data_model)
#类中每一个测试用例运行后都会执行，用例的释放
    def tearDown(self):
        print('tearDown...')

    def test_bo(self):
        print("测试开始：")
        # data_model = self.data_model
        # optimize_cnn(data_model)

    def test_train(self):
        stat_dict=self.stat_dict
        model_data = self.data_model
        model_dict = model_data.data_source.data_config.model_dict
        # bo_json_file = os.path.join(model_dict["dir"]["Out"], "bo_logs.json")
        # bo_json = unserialize_bo_json(bo_json_file)
        # print(bo_json)
        # n_input = bo_json["params"]["n_input"]
        # batch_size = bo_json["params"]["batch_size"]
        # epochs = bo_json["params"]["epochs"]
        # 首先要对数据进行离散化处理，否则后面会报错
        # n_input = int(n_input)
        # batch_size = int(batch_size)
        # epochs = int(epochs)
        # n_input = int(14)
        # batch_size = int(40)
        # epochs = int(82)
        n_input = int(14)
        batch_size = int(100)
        epochs = int(100)
        # 调用to_supervised将数据分为输入和输出部分
        data, targets = to_supervised(model_data.load_data(model_dict), n_input)
        model_trained = cnn_train(input=data, output=targets, batch_size=batch_size, epochs=epochs, model_dict=model_dict,stat_dict=stat_dict)
        # cnn_pred_value, cnn_obs_value = cnn_train_forecast(input=data, output=targets, stat_dict)


        print(model_trained)







if __name__ == '__main__':
    unittest.main()  #运行类中每一个测试用例

