import os
import unittest
import definitions
from data import *
from utils import *


class MyTestCase(unittest.TestCase):
    def test_data_model(self):
        print('Starting ...')

        configFile = definitions.CONFIG_FILE
        # 读取模型配置文件
        configData = DataConfig(configFile)
        # 准备训练数据
        sourceData = DataSource(configData, configData.model_dict["data"]["tRangeTrain"])
        # 构建输入数据类对象
        dataModel = DataModel(sourceData)
        # 序列化保存对象
        dir_temp = sourceData.all_configs["temp_dir"]
        source_file = os.path.join(dir_temp, 'data_source.txt')
        stat_file = os.path.join(dir_temp, 'Statistics.json')
        flow_file = os.path.join(dir_temp, 'flow')
        t_s_dict_file = os.path.join(dir_temp, 'dictTimeSpace.json')

        # 存储data_model，因为data_model里的数据如果直接序列化会比较慢，所以各部分分别序列化，dict的直接序列化为json文件，数据的HDF5
        serialize_pickle(sourceData, source_file)
        serialize_json(dataModel.stat_dict, stat_file)
        serialize_numpy(dataModel.data_flow, flow_file)
        serialize_json(dataModel.t_s_dict, t_s_dict_file)


if __name__ == '__main__':
    unittest.main()
