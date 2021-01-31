import numpy as np
import scipy.stats


def statError(target, pred):
    ngrid, nt = pred.shape
    # Bias
    Bias = np.nanmean(pred - target, axis=1)
    # RMSE
    RMSE = np.sqrt(np.nanmean((pred - target) ** 2, axis=1))
    # ubRMSE
    predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
    targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
    predAnom = pred - predMean
    targetAnom = target - targetMean
    ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom) ** 2, axis=1))
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
            yymean = yy.mean()
            SST = np.sum((yy - yymean) ** 2)
            SSReg = np.sum((xx - yymean) ** 2)
            SSRes = np.sum((yy - xx) ** 2)
            R2[k] = 1 - SSRes / SST
            NSE[k] = 1 - SSRes / SST
            # FHV the peak flows bias 2%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3 * len(pred_sort))
            indexhigh = round(0.98 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
            PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
            outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, R2=R2, NSE=NSE, FLV=PBiaslow, FHV=PBiashigh)
    return outDict


def cal_4_stat_inds(b):
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def cal_stat(x):
    a = x.flatten()
    b = a[~np.isnan(a)]
    return cal_4_stat_inds(b)


def cal_stat_gamma(x):
    """for daily streamflow and precipitation"""
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(np.sqrt(b) + 0.1)  # do some tranformation to change gamma characteristics
    return cal_4_stat_inds(b)


def cal_stat_basin_norm(x, basinarea, meanprep):
    """for daily streamflow normalized by basin area and precipitation
    basinarea = readAttr(gageDict['id'], ['area_gages2'])
    meanprep = readAttr(gageDict['id'], ['p_mean'])
    """
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    temparea = np.tile(basinarea, (1, x.shape[1]))
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    flowua = (x * 0.0283168 * 3600 * 24) / (
            (temparea * (10 ** 6)) * (tempprep * 10 ** (-3)))  # unit (m^3/day)/(m^3/day)
    return cal_stat_gamma(flowua)


def trans_norm(x, var_lst, stat_dict, *, to_norm):
    """normalization; when to_norm=False, anti-normalization
    :parameter
        x：ad or 3d
            2d：1st dim is gauge，2nd dim is var type
            3d：1st dim is gauge，2nd dim is time，3rd dim is var type
    """
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.zeros(x.shape)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                if var == 'prcp' or var == 'usgsFlow':
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var == 'prcp' or var == 'usgsFlow':
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var == 'prcp' or var == 'usgsFlow':
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var == 'prcp' or var == 'usgsFlow':
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
    return out


def basin_norm(x, basin_area, mean_prep, to_norm):
    """for regional training, gageid should be numpyarray"""
    nd = len(x.shape)
    # meanprep = readAttr(gageid, ['q_mean'])
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basin_area, (1, x.shape[1]))
    tempprep = np.tile(mean_prep, (1, x.shape[1]))
    if to_norm is True:
        flow = (x * 0.0283168 * 3600 * 24) / ((temparea * (10 ** 6)) * (tempprep * 10 ** (-3)))  # (m^3/day)/(m^3/day)
    else:
        flow = x * ((temparea * (10 ** 6)) * (tempprep * 10 ** (-3))) / (0.0283168 * 3600 * 24)
    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow
