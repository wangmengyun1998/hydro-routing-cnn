import definitions
from bayes_opt.util import Colours
from data import *
from hydroDL import *

print('Starting ...')

configFile = definitions.CONFIG_FILE
# 读取模型配置文件
configData = DataConfig(configFile)
# 准备训练数据
sourceData = DataSource(configData, configData.model_dict["data"]["tRangeTrain"])
# 构建输入数据类对象
dataModel = DataModel(sourceData)

# 进行模型训练
# train model
print(Colours.yellow("--- Optimizing CNN ---"))
optimize_cnn(dataModel)
