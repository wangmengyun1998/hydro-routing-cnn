"""read config from config_file"""
import collections
import os
from collections import OrderedDict
from configparser import ConfigParser
import definitions  # this line is needed, Please don't delete it


def update(opt, **kw):
    for key in kw:
        if key in opt:
            try:
                opt[key] = type(opt[key])(kw[key])
            except ValueError:
                print('skiped ' + key + ': wrong type')
        else:
            print('skiped ' + key + ': not in argument dict')
    return opt

def name_pred(m_dict, out, t_range, epoch, subset=None, suffix=None):
    """训练过程输出"""
    loss_name = m_dict['loss']['name']
    file_name = '_'.join([str(t_range[0]), str(t_range[1]), 'ep' + str(epoch)])
    if loss_name == 'SigmaLoss':
        file_name = '_'.join('SigmaX', file_name)
    if suffix is not None:
        file_name = file_name + '_' + suffix
    file_path = os.path.join(out, file_name + '.csv')
    return file_path


def init_path(config_file):
    """根据配置文件读取数据源路径"""
    cfg = ConfigParser()
    cfg.read(config_file,encoding="UTF-8")  #读取config文件
    sections = cfg.sections() #config里面[basic]、[sanxia]、[model]
    data_input = cfg.get(sections[0], 'download') #download = data
    data_output = cfg.get(sections[0], 'output') #output = output
    data_temp = cfg.get(sections[0], 'temp') #temp = temp
    root = eval(cfg.get(sections[0], 'prefix')) #prefix = os.path.join(definitions.ROOT_DIR,"example")
    data_input = os.path.join(root, data_input)  #创建 F:\科研类\code\hydro-ideas-dl\example\data
    data_output = os.path.join(root, data_output)#创建F:\科研类\code\hydro-ideas-dl\example\output
    data_temp = os.path.join(root, data_temp)#创建F:\科研类\code\hydro-ideas-dl\example\temp
    if not os.path.isdir(data_input):
         os.makedirs(data_input)    #创建data文件夹
    if not os.path.isdir(data_output):
         os.makedirs(data_output)  #创建output文件夹
    if not os.path.isdir(data_temp):
        os.makedirs(data_temp)  #创建temp文件夹
    path_data = collections.OrderedDict(
        DB=os.path.join(data_input, cfg.get(sections[0], 'data')),
       Out=os.path.join(data_output, cfg.get(sections[0], 'data')),
       Temp=os.path.join(data_temp, cfg.get(sections[0], 'data')))  #OrderedDict([('DB', 'F:\\科研类\\code\\hydro-ideas-dl\\example\\data\\sanxia'), ('Out', 'F:\\科研类\\code\\hydro-ideas-dl\\example\\output\\sanxia'), ('Temp', 'F:\\科研类\\code\\hydro-ideas-dl\\example\\temp\\sanxia')])
    if not os.path.isdir(path_data["DB"]):
         os.makedirs(path_data["DB"])  #创建sanxia文件夹
    if not os.path.isdir(path_data["Out"]):
        os.makedirs(path_data["Out"])       #创建sanxia文件夹
    if not os.path.isdir(path_data["Temp"]):
       os.makedirs(path_data["Temp"])    #创建sanxia文件夹
    return path_data



class DataConfig(object):
    def __init__(self, config_file):
        self.config_file = config_file
        self.data_path = init_path(config_file)
        opt_data, opt_train = self.init_model_param()
        self.model_dict = OrderedDict(dir=self.data_path, data=opt_data, train=opt_train)


    def init_data_param(self):
        """read camels or gages dataset configuration
                根据配置文件读取有关输入数据的各项参数"""
        config_file = self.config_file
        cfg = ConfigParser()
        cfg.read(config_file,encoding="UTF-8")
        sections = cfg.sections()
        section = cfg.get(sections[0], 'data')
        options = cfg.options(section)
        # streamflow
        t_range_all = eval(cfg.get(section, options[0]))
        sites = eval(cfg.get(section, options[1]))

        opt_data = collections.OrderedDict(tRangeAll=t_range_all, sites=sites)

        return opt_data

    def init_model_param(self):
        """根据配置文件读取有关模型的各项参数，返回optModel, optLoss, optTrain三组参数，分成几组的原因是为写成json文件时更清晰"""
        config_file = self.config_file
        cfg = ConfigParser()
        cfg.read(config_file,encoding="UTF-8")
        section = 'model'
        options = cfg.options(section)
        # train and test time range
        t_range_bo = eval(cfg.get(section, options[0]))
        t_range_train = eval(cfg.get(section, options[1]))
        t_range_test = eval(cfg.get(section, options[2]))
        # data processing parameter
        rm_nan = eval(cfg.get(section, options[3]))
        opt_data = collections.OrderedDict(tRangeBo=t_range_bo, tRangeTrain=t_range_train, tRangeTest=t_range_test,
                                           rmNan=rm_nan)

        # model parameters. 首先读取几个训练使用的基本模型参数，主要是epoch和batch
        batch_range = eval(cfg.get(section, options[4]))
        epoch_range = eval(cfg.get(section, options[5]))
        input_range = eval(cfg.get(section, options[6]))
        opt_train = collections.OrderedDict(batchRange=batch_range, epochRange=epoch_range, inputRange=input_range)

        return opt_data, opt_train
    def read_data_config(self):
        dir_db = self.data_path.get("DB")
        dir_out = self.data_path.get("Out")
        dir_temp = self.data_path.get("Temp")

        data_params = self.init_data_param()

        # 径流数据配置
        flow_gage_time = data_params.get("tRangeAll")
        flow_gage_name = data_params.get("sites")
        flow_gage_file = []
        for i in range(len(flow_gage_name)):
            if flow_gage_name[i] == "lijiawan":
                t_range_temp = "1951-2007"
            elif flow_gage_name[i] == "yichang":
                t_range_temp = "1952-2008"
            elif flow_gage_name[i] == "wulong":
                t_range_temp = "1950-2006"
            else:
                t_range_temp = "1950-2007"
            flow_gage_file_temp = "-".join([str(i + 1), flow_gage_name[i], t_range_temp, "day", "runoff.csv"])
            flow_gage_file.append(os.path.join(dir_db, flow_gage_file_temp))

        return collections.OrderedDict(root_dir=dir_db, out_dir=dir_out, temp_dir=dir_temp,
                                       flow_gage_time=flow_gage_time, flow_gage_name=flow_gage_name,
                                       flow_gage_file=flow_gage_file)

# if __name__ == '__main__':
#     config_file = "F:/科研类/code/hydro-ideas-dl/example/User/config.ini"
#     DC1 = DataConfig(config_file)
#     print(DC1.config_file)
#     print(DC1.data_path)
#     print(DC1.init_data_param())
#     print(DC1.init_model_param())
#     print(DC1.model_dict)


