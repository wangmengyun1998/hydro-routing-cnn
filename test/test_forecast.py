import os
import unittest
import torch
from bayes_opt.util import Colours
from hydroDL.model.cnn import TryCnn
import definitions
from data import *
from hydroDL.master import routing_cnn, cnn_test, evaluate_forecasts
from utils import *
from hydroDL import *
from visual import *
import shap
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np




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
    shap_values_days_states_file=os.path.join(dir_temp,"shap_values_days_states")
    shap_values_days_states_npy = os.path.join(dir_temp, "shap_values_days_states.npy")
    shap_days_file = os.path.join(dir_temp, " shap_days")
    shap_days_npy = os.path.join(dir_temp, " shap_days.npy")
    shap_stations_file = os.path.join(dir_temp, " shap_stations")
    shap_stations_npy = os.path.join(dir_temp, " shap_stations.npy")
    shap_values_stations_days_file = os.path.join(dir_temp, "  shap_values_stations_days")
    shap_values_stations_days_npy = os.path.join(dir_temp, "  shap_values_stations_days.npy")
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
        # model_file = os.path.join(model_dict["dir"]["Out"], "model.yaml")
        # weight_file = os.path.join(model_dict["dir"]["Out"], "weights.h5")
        # bo_json_file = os.path.join(model_dict["dir"]["Out"], "bo_logs.json")
        # bo_json = unserialize_bo_json(bo_json_file)
        # n_input = int(bo_json["params"]["n_input"])
        n_input = int(14)
        try_cnn = TryCnn()
        data, targets = routing_cnn.to_supervised(model_data.load_data(model_dict), n_input)
        obs_value, pred_value = cnn_test(try_cnn,X=data, Y=targets, stat_dict=stat_dict)
        # obs_value, pred_value = cnn_test(data, targets, stat_dict, model_file=model_file, weight_file=weight_file)
        print("the observe value:", obs_value)
        print("the predict value:", pred_value)
        serialize_numpy(obs_value, self.obs_file)
        serialize_numpy(pred_value, self.pred_file)


    def test_shap(self):
        print(Colours.yellow("--- CNN shap---"))
        source_data = unserialize_pickle(self.data_source_dump)
        stat_dict = unserialize_json(self.stat_file)
        data_flow = unserialize_numpy(self.flow_file)
        t_s_dict = unserialize_json(self.t_s_dict_file)
        model_data = DataModel(source_data, data_flow, t_s_dict, stat_dict)
        model_dict = model_data.data_source.data_config.model_dict
        n_input = int(14)
        try_cnn = TryCnn()
        data, targets = routing_cnn.to_supervised(model_data.load_data(model_dict), n_input)
        try_cnn.load_state_dict(torch.load("F:\科研类\codes\hydro-routing-cnn\checkpoint.pt"))
        # x1 = data.reshape(715, 1, -1)
        # x = x1.reshape(-1, 1, 1, x1.shape[2])
        x = data.reshape(-1,1, data.shape[1], data.shape[2])
        x = torch.from_numpy(x).float()
        try_cnn.eval()
        # x_pred = try_cnn(x[301:306])
        # print(x[301:306])
        # print(x_pred)
        print("======计算SHAP========")
        # 新建一个解释器（模型，数据）
        background = x[np.random.choice(x.shape[0], 100, replace=False)]
        e = shap.DeepExplainer(try_cnn,background)
        # e = shap.DeepExplainer(try_cnn, x)
        shap_values= e.shap_values(x)
        shap_values_stations_days=np.abs(shap_values).sum(axis=0).reshape(14,data.shape[2])
        shap_days=shap_values_stations_days.sum(axis=1)
        shap_stations = shap_values_stations_days.sum(axis=0)
        # 计算base_line
        # y_base = e.expected_value
        # print("y_base的值：", y_base)
        # print("y_base+shap值的和：",y_base+shap_values.sum())
        shap_values_array=shap_values.reshape(-1, data.shape[1]*data.shape[2])
        shap_arrays_values=[]
        for i in range(shap_values_array.shape[0]):
            new_array = np.zeros((shap_values_array.shape[0] - 1) * data.shape[2])
            if i==0:
                ndarray=np.append(shap_values_array[i],new_array)
            elif i==shap_values_array.shape[0]-1:
                ndarray = np.insert(shap_values_array[i], 0, new_array)
            else:
                ndarray = np.pad(shap_values_array[i], (i * data.shape[2], (shap_values_array.shape[0] - 1 - i) * data.shape[2]),
                                 'constant')
            shap_arrays_values.append(ndarray)
        shap_arrays_values=np.array(shap_arrays_values)
        shap_arrays_values_abs=np.abs(shap_arrays_values).sum(axis=0).reshape(-1,data.shape[2])
        print(shap_arrays_values_abs)
        shap_values_days_stations=[]
        for j in range(shap_arrays_values_abs.shape[0]):
             if j<14:
                 shap_values_day_state=shap_arrays_values_abs[j]/(j+1)
             elif j>=shap_arrays_values_abs.shape[0]-14:
                 shap_values_day_state = shap_arrays_values_abs[j] / (shap_arrays_values_abs.shape[0]-j)
             else:
                 shap_values_day_state = shap_arrays_values_abs[j] / 14
             shap_values_days_stations.append(shap_values_day_state)
        shap_values_days_stations = np.array(shap_values_days_stations)
        print( shap_values_days_stations)
        serialize_numpy(shap_values_days_stations,self.shap_values_days_states_file)
        serialize_numpy( shap_days, self. shap_days_file)
        serialize_numpy(shap_stations, self.shap_stations_file)
        serialize_numpy(shap_values_stations_days, self. shap_values_stations_days_file)



    def test_plot(self):
        shap_values_stations_days = unserialize_numpy(self. shap_values_stations_days_npy)
        shap_days=unserialize_numpy(self.shap_days_npy)
        shap_stations = unserialize_numpy(self.shap_stations_npy)
        shap_values_days_states = unserialize_numpy(self.shap_values_days_states_npy)
        observe_flow = unserialize_numpy(self.obs_file_npy)
        predict_flow = unserialize_numpy(self.pred_file_npy)
        #画出observe_flow、predict_flow曲线
        plt.plot(observe_flow )
        plt.plot(predict_flow)
        plt.legend(['observe ', 'predict'], loc='upper left')
        plt.show()


        if shap_values_stations_days.shape[1]==6:
            columns=["pingshan", "gaochang", "lijiawan", "beibei", "wulong","yichang"]
            index = ["pingshan", "gaochang", "lijiawan", "beibei", "wulong","yichang"]
        else:
            columns = ["pingshan", "gaochang", "lijiawan", "beibei", "wulong"]
            index = ["pingshan", "gaochang", "lijiawan", "beibei", "wulong"]
        # 可视化每一天各站的情况
        shap_values_stations_days_pd = pd.DataFrame(shap_values_stations_days, index=np.arange(1, 15, 1),
                                                    columns=columns)
        shap_values_stations_days_pd_hotmap=shap_values_stations_days_pd
        # 可视化14天
        shap_days_pd = pd.DataFrame(shap_days, index=np.arange(1, 15, 1))
        # 可视化6个站
        shap_stations_pd = pd.DataFrame(shap_stations, index=index)
        # 可视化各站每一天热点图
        shap_values_days_states_pd = pd.DataFrame(shap_values_days_states,
                                                  columns=columns)
        shap_values_stations_days_pd.plot(kind='bar', rot=60, legend=7, fontsize=8, width=0.5)
        shap_days_states_ts = shap_values_days_states_pd.loc[14:364]
        shap_days_states_te = shap_values_days_states_pd.loc[365:shap_values_days_states_pd.shape[0]-14]
       #创建图形figure1
        fig1=plt.figure(figsize=(100,100))
        grid=plt.GridSpec(4,7,hspace=0.7,wspace=0.7)
        ax1=fig1.add_subplot(grid[:2,:2])
        shap_days_pd.plot(kind='bar',rot=60, legend=False, ax=ax1,fontsize = 8)
        ax1.set_ylabel("shap_values")
        ax2 = fig1.add_subplot(grid[:2, 2:4])
        shap_stations_pd.plot(kind='bar', rot=-60, legend=False,ax=ax2,fontsize = 5)
        ax3 = fig1.add_subplot(grid[:2, 4:])
        sns.heatmap(shap_values_stations_days_pd_hotmap.stack().unstack(0), cmap="YlGnBu", cbar=True, ax=ax3)
        ax3.set_yticklabels(columns, rotation=60, fontsize=8)
        ax3.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12,13,14],rotation=60, fontsize=8)
        ax4 = fig1.add_subplot(grid[2:, 0:3])
        sns.heatmap(shap_days_states_ts.stack().unstack(0), cmap="YlGnBu",cbar=True,ax=ax4)
        ax4.set_yticklabels(columns, rotation=60,fontsize = 8)
        ax4.set_ylabel("stations")
        ax4.set_xticklabels([])
        ax4.set_xlabel('the year of 2006')
        ax5 = fig1.add_subplot(grid[2:, 4:])
        sns.heatmap(shap_days_states_te.stack().unstack(0), cmap="Blues", cbar=True, ax=ax5)
        ax5.set_yticklabels(columns, rotation=60, fontsize=8)
        ax5.set_xticklabels([])
        ax5.set_xlabel('the year of 2007')

        # plt.tight_layout()
        #图形figure2
        if shap_values_stations_days_pd.shape[1]==5:
            fig2 = plt.figure(figsize=(100, 100))
            grid = plt.GridSpec(4, 3, hspace=0.7, wspace=0.7)
            ax1 = fig2.add_subplot(grid[:2, 0])
            shap_values_stations_days_pd.iloc[:,0].plot(title="pingshan",kind='bar',rot=60, legend=False, ax=ax1,fontsize = 8)
            ax2= fig2.add_subplot(grid[:2,1])
            shap_values_stations_days_pd.iloc[:,1].plot(title="gaochang",kind='bar', rot=60, legend=False, ax=ax2, fontsize=8)
            ax3 = fig2.add_subplot(grid[:2, 2])
            shap_values_stations_days_pd.iloc[:,2].plot(title="lijiawan",kind='bar', rot=60, legend=False, ax=ax3, fontsize=8)
            ax4 = fig2.add_subplot(grid[2:, 0])
            shap_values_stations_days_pd.iloc[:,3].plot(title="beibei",kind='bar', rot=60, legend=False, ax=ax4, fontsize=8)
            ax5 = fig2.add_subplot(grid[2:, 1])
            shap_values_stations_days_pd.iloc[:,4].plot(title="wulong",kind='bar', rot=60, legend=False, ax=ax5, fontsize=8)
        else:
            fig2 = plt.figure(figsize=(100, 100))
            grid = plt.GridSpec(4, 3, hspace=0.7, wspace=0.7)
            ax1 = fig2.add_subplot(grid[:2, 0])
            shap_values_stations_days_pd.iloc[:, 0].plot(title="pingshan", kind='bar', rot=60, legend=False, ax=ax1,fontsize=8)
            ax2 = fig2.add_subplot(grid[:2, 1])
            shap_values_stations_days_pd.iloc[:, 1].plot(title="gaochang", kind='bar', rot=60, legend=False, ax=ax2,fontsize=8)
            ax3 = fig2.add_subplot(grid[:2, 2])
            shap_values_stations_days_pd.iloc[:, 2].plot(title="lijiawan", kind='bar', rot=60, legend=False, ax=ax3,fontsize=8)
            ax4 = fig2.add_subplot(grid[2:, 0])
            shap_values_stations_days_pd.iloc[:, 3].plot(title="beibei", kind='bar', rot=60, legend=False, ax=ax4,fontsize=8)
            ax5 = fig2.add_subplot(grid[2:, 1])
            shap_values_stations_days_pd.iloc[:, 4].plot(title="wulong", kind='bar', rot=60, legend=False, ax=ax5, fontsize=8)
            ax6 = fig2.add_subplot(grid[2:, 2])
            shap_values_stations_days_pd.iloc[:, 5].plot(title="yichang", kind='bar', rot=60, legend=False, ax=ax6,fontsize=8)
        # #可视化各站每一天
        #取出前一年的数据
        # # dates_ts=pd.period_range("1 1 1998", "12 31 1998", freq='D')
        # shap_days_states_ts=shap_values_days_states_pd.loc[:364]
        # # #取出后一年的数据
        # # # dates_te= pd.date_range("1 1 1999", periods=shap_values_days_states_pd.loc[365:].shape[0], freq="D")
        # shap_days_states_te = shap_values_days_states_pd.loc[365:shap_values_days_states_pd.shape[0]-2]
        # # print(shap_values_days_states_pd)
        # #画出全年的hotmap图
        # f, (ax1, ax2) = plt.subplots(figsize=(10, 10), nrows=2)
        # # ax1.imshow(np.array(shap_days_states_ts),cmap="YlGnBu")
        # sns.heatmap(shap_days_states_ts.stack().unstack(0), cmap="YlGnBu",cbar=True,ax=ax1)
        # ax1.set_title('the year of 2000')
        # ax1.set_ylabel('')
        # ax1.set_yticklabels(columns,rotation=30)
        # ax1.set_xlabel('')
        # ax1.set_xticklabels([])
        # # ax2.imshow(np.array(shap_days_states_te),cmap="Blues")
        # sns.heatmap(shap_days_states_te.stack().unstack(0),cmap="Blues",ax=ax2)
        # ax2.set_title('the year of 2001')
        # ax2.set_yticklabels(columns,rotation=30)
        # ax2.set_xlabel('')
        # ax2.set_xticklabels([])
        # ax2.set_xlabel('day')
       #展示图像
        plt.show()
        # #画出全年每月的hotmap
        # fig1, axes = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(20, 50))
        # xlabels=["pingshan", "gaochang", "lijiawan", "beibei", "wulong",
        #                                                    "yichang"]
        # fig1.subplots_adjust(wspace=0.5, hspace=0.1)
        # for i,row in enumerate(axes):
        #     for j, col in enumerate(row):
        #         if i==0 and j==0:
        #             # col.set_xticks(np.arange(0,6,1),["pingshan", "gaochang", "lijiawan", "beibei", "wulong",
        #             #                                        "yichang"])
        #             # col.set_yticks(range(len(shap_days_states_ts.loc[:30])))
        #             # col.set_xlabels(xlabels)
        #             im=col.imshow(np.array(shap_days_states_ts.loc[:30]),cmap="Blues",origin='lower',aspect='auto',interpolation = 'none')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Jan')
        #
        #         elif i==0 and j==1:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[31:58]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Feb')
        #         elif i == 0 and j == 2:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[59:89]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Mar')
        #         elif i == 1 and j == 0:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[90:119]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Apr')
        #         elif i == 1 and j == 1:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[120:150]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('May')
        #         elif i == 1 and j == 2:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[151:180]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Jun')
        #         elif i == 2 and j == 0:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[181:211]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Jul')
        #         elif i == 2 and j == 1:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[212:242]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Aug')
        #         elif i == 2 and j == 2:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[243:272]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Sep')
        #         elif i == 3 and j == 0:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[273:303]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Oct')
        #         elif i == 3 and j == 1:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[304:333]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Nov')
        #         elif i == 3 and j == 2:
        #             im = col.imshow(np.array(shap_days_states_ts.loc[334:364]),cmap="Blues",origin='lower',aspect='auto')
        #             ax_cb = fig1.colorbar(im, ax=col,shrink=0.5)
        #             if col.is_last_row():
        #                 col.set_xlabel('x')
        #             if col.is_first_col():
        #                 col.set_ylabel('Dec')
        # # cb.ax.tick_params()
        # # cb.set_label("colorbar")
        # fig1.text(0.5, 0, 'station', ha='center')
        # fig1.text(0, 0.5, 'month', va='center', rotation='vertical')
        # plt.tight_layout()
        inds_dict = evaluate_forecasts(observe_flow, predict_flow)
        print(inds_dict)
        # plot_box_inds(subset_of_dict(inds_dict, ["RMSE"]))
        # plot_box_inds(subset_of_dict(inds_dict, ["QR", "R2"]))
        #plot_box_inds(subset_of_dict(inds_dict, ["NSE", "QR", "R2"]))

if __name__ == '__main__':
    unittest.main()

