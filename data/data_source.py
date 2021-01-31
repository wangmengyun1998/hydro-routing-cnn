"""get data from data source -- 6 csv tables"""
import numpy as np
import pandas as pd

from utils import *
from data.data_config import *
from definitions import CONFIG_FILE


class DataSource(object):
    """Directly read data"""

    def __init__(self, config_data, t_range):
        """read configuration of data source. 读取配置，准备数据，关于数据读取部分，可以放在外部需要的时候再执行"""
        self.data_config = config_data
        self.all_configs = config_data.read_data_config()
        # t_range: 训练数据还是测试数据，需要外部指定
        self.t_range = t_range
        #'tRangeTrain', ['1995-01-01', '1997-01-01']


    def read_streamflow(self):
        """read all streamflow data"""
        # 先把原来excel的数据读取到矩阵中，前面5个站的格式都是以日为行，以年为列，最后一个站的数据是一二三列为年月日，最后一列是数据
        # 读取之后，逐步处理成以站点为行，日期为列，列数暂定14，的矩阵。先统一一下数据格式，每个站点的数据都转变成一行，从最早的日期开始，到最后的日期。
        nt = hydro_time.t_range_days(self.t_range).shape[0]
        sites_lst = self.all_configs["flow_gage_name"]
        y = np.empty([len(sites_lst), nt])
        for k in range(len(sites_lst)):
            data_obs = self.read_flow_site(sites_lst[k])
            y[k, :] = data_obs
        return y

    def read_flow_site(self, site_name):
        site_index = 0
        for i in range(len(self.all_configs["flow_gage_name"])):
            if site_name == self.all_configs["flow_gage_name"][i]:
                site_index = i
                break
        site_file = self.all_configs["flow_gage_file"][site_index]
        data_temp = pd.read_csv(site_file,header=None, encoding="UTF-8")
        #data_temp = pd.read_csv(site_file, header=None)
        t_range_list = hydro_time.t_range_days(self.t_range)
        nt = t_range_list.shape[0]
        data_obs = np.full([nt], np.nan)
        date_columns = ['year', 'month', 'day']
        if 'yichang' == site_name:
            df_date = data_temp[[0, 1, 2]]
            df_date.columns = date_columns
            date = pd.to_datetime(df_date).values.astype('datetime64[D]')
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            df_value = data_temp.iloc[:, 3]
            data_obs[ind2] = df_value[ind1]
        else:
            all_start_year = int(site_file.split("\\")[-1].split("-")[2])
            all_end_year = int(site_file.split("\\")[-1].split("-")[3])
            # all_end_year is included, so +1 in np.arange function
            all_years = np.arange(all_start_year, all_end_year + 1)
            new_column_id_vars = ["month_day"]
            new_columns_value_vars = [str(temp_year) for temp_year in all_years]
            new_columns = new_column_id_vars.copy()
            [new_columns.append(value_var) for value_var in new_columns_value_vars]
            data_temp.columns = new_columns
            flow_col_name = "flow"
            data_new = pd.melt(data_temp, id_vars=new_column_id_vars, value_vars=new_columns_value_vars,
                               var_name='year', value_name=flow_col_name)
            years = data_new.iloc[:, 1].values.astype(int)
            df_month_day = data_new.iloc[:, 0]
            months = np.array([int(month_day.split('月')[0]) for month_day in df_month_day])
            days = np.array([int(month_day.split('月')[1][:-1]) for month_day in df_month_day])
            df_date = pd.DataFrame({date_columns[0]: years, date_columns[1]: months, date_columns[2]: days})
            date = pd.to_datetime(df_date).values.astype('datetime64[D]')
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            df_value = data_new[flow_col_name]
            data_obs[ind2] = df_value[ind1]
        return data_obs

    def screen_streamflow(self, streamflow, usgs_ids=None, time_range=None):
        """ choose appropriate ones from all usgs sites
            Parameters
            ----------
            streamflow : numpy ndarray -- all usgs sites(config)' data, its index are 'sites', its columns are 'day',
                                   if there is some missing value, usgs should already be filled by nan
            usgs_ids: list -- chosen sites' ids
            time_range: list -- chosen time range

            Returns
            -------
            usgs_out : ndarray -- streamflow  1d-var is gage, 2d-var is day
            sites_chosen: [] -- ids of chosen gages

            Examples
            --------
            usgs_screen(usgs, ["02349000","08168797"], [‘1995-01-01’,‘2015-01-01’])
        """
        sites_chosen = np.zeros(streamflow.shape[0])
        # choose the given sites
        usgs_all_sites = self.all_configs["flow_gage_name"]
        if usgs_ids:
            sites_index = np.where(np.in1d(usgs_ids, usgs_all_sites))[0]
            sites_chosen[sites_index] = 1
        else:
            sites_index = np.arange(streamflow.shape[0])
            sites_chosen = np.ones(streamflow.shape[0])
        # choose data in given time range
        all_t_list = hydro_time.t_range_days(self.t_range)
        t_lst = all_t_list
        if time_range:
            # calculate the day length
            t_lst = hydro_time.t_range_days(time_range)
        ts, ind1, ind2 = np.intersect1d(all_t_list, t_lst, return_indices=True)

        streamflow_temp = streamflow[sites_index]  # 先取出想要的行数据
        usgs_values = streamflow_temp[:, ind1]  # 再取出要求的列数据

        # get discharge data of chosen sites, and change to ndarray
        usgs_out = usgs_values[np.where(sites_chosen > 0)]
        gages_chosen_id = [usgs_all_sites[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0]

        return usgs_out, gages_chosen_id, ts


