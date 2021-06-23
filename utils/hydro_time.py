"""补充时间处理相关函数"""
import datetime as dt,datetime
import numpy as np




def t2dt(t, hr=False):
    t_out = None
    if type(t) is int:
        if 30000000 > t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            t_out = t if hr is False else t.datetime()

    if type(t) is dt.date:
        t_out = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        t_out = t.date() if hr is False else t

    if t_out is None:
        raise Exception('hydroDL.utils.t2dt failed')
    return t_out


def t_range2_array(t_range, *, step=np.timedelta64(1, 'D')):
    sd = t2dt(t_range[0])
    ed = t2dt(t_range[1])
    tArray = np.arange(sd, ed, step)
    return tArray


#def t_range_days(t_range, *, step=np.timedelta64(1, 'D')):
def t_range_days(t_range):#用于控制时间间隔，间隔多少年/天/时/分/秒

    """将给定的一个区间，转换为每日一个值的数组"""
    # sd = dt.datetime.strptime(t_range[0], '%Y-%m-%d')
    # ed = dt.datetime.strptime(t_range[1], '%Y-%m-%d')
    # # arange函数结果是左闭右开区间
    # t_array = np.arange(sd, ed, step)
    # return t_array
    if len(t_range)>2:
        sd1 = dt.datetime.strptime(t_range[0], '%Y-%m-%d')
        ed1 = dt.datetime.strptime(t_range[1], '%Y-%m-%d')
        group1=np.arange(sd1, ed1, dtype='datetime64[D]')
        sd2 = dt.datetime.strptime(t_range[2], '%Y-%m-%d')
        ed2 = dt.datetime.strptime(t_range[3], '%Y-%m-%d')
        group2 = np.arange(sd2, ed2,dtype='datetime64[D]')
        sd3 = dt.datetime.strptime(t_range[4], '%Y-%m-%d')
        ed3 = dt.datetime.strptime(t_range[5], '%Y-%m-%d')
        group3 = np.arange(sd3, ed3,dtype='datetime64[D]')
        sd4 = dt.datetime.strptime(t_range[6], '%Y-%m-%d')
        ed4 = dt.datetime.strptime(t_range[7], '%Y-%m-%d')
        group4 = np.arange(sd4, ed4, dtype='datetime64[D]')
        t_array=np.concatenate((group1, group2,group3,group4))
    else:
        sd1 = dt.datetime.strptime(t_range[0], '%Y-%m-%d')
        ed1 = dt.datetime.strptime(t_range[1], '%Y-%m-%d')
        group1=np.arange(sd1, ed1, dtype='datetime64[D]')
        t_array=group1
    return t_array


# t_range=["1998-01-01","1999-01-01"]
# a=t_range_days(t_range)
# print(list(a))
# print(a.shape[0])
# print(a.shape)







    #t_array=np.append(group1,group2,group3,group4)


def t_range_years(t_range):
    start_year = int(t_range[0].split("-")[0])
    end_year = int(t_range[1].split("-")[0])
    year_range_list = np.arange(start_year, end_year)
    return year_range_list


def get_year(a_time):
    """返回时间的年份"""
    if isinstance(a_time, datetime.date):
        return a_time.year
    elif isinstance(a_time, np.datetime64):
        return a_time.astype('datetime64[Y]').astype(int) + 1970
    else:
        return int(a_time[0:4])


def intersect(t_lst1, t_lst2):
    C, ind1, ind2 = np.intersect1d(t_lst1, t_lst2, return_indices=True)
    return ind1, ind2
