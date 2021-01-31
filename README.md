# hydro-ideas-dl

这是一个尝试水文水资源领域相关问题的 deeep learning 解决方法的repo。

主要记录相关问题的由来，以及用deep learning 方法来分析的动机。

也给出编程所需基本环境配置和简单框架。

## 问题列表

本节列出的问题有一些相对清晰的基本思路，可供动手实践，是否是low-hanging fruits 暂不确定。

问题不分先后，是日常积累时间顺序。

- 河道汇流
- 人类活动对流域径流预测的影响
- 使用更多遥感数据来帮助有水库流域的径流预测
- 径流预测中初始状态的影响

### 河道汇流-CNN

在水文领域，实践中人们计算河道汇流时，通常使用的是马斯京根法，此公式本身是属于有一定物理意义的经验公式，计算并不精确，而比它精确计算的公式往往面临效率问题。
因此科研人员有时会依赖一些计算效率高的黑箱算法，比如ANN神经网络。

另一方面，水文中存在很多经验公式，有些公式本身就是从数据中总结的，因此如果有足够的观测数据，machine learning可能帮助我们发现更符合数据规律的方程。

在坡面汇流计算时，常用的经典算法是单位线，作为一种基于系统分析的算法，卷积计算在其中扮演了十分重要的角色，而河道汇流有时也可以使用该算法。

受启发于此，考虑使用卷积神经网络来尝试开展河道汇流计算，看看相比于ANN+筛选的做法，能不能在计算效率或精度上有所提升。

此idea了参考Chen等的[论文](https://ascelibrary.org/doi/abs/10.1061/(ASCE)HE.1943-5584.0000932)。
这篇论文用copula熵筛选计算因子，最后能筛选出前面某个站第几天的数据来输入ANN比较好。现在我们通过卷积，直接利用卷积网络来做此事。

具体来说，把每个站各自时间序列当作一个维度，把各个站共同作为一个空间序列的维度，形成一种二维表达，类比于图片。
试试cnn有没有效果，以每个时段为一个样本，来训练多个卷积核。

神经网络和计算公式之间本身的联系还可以进一步参考以下文献来分析：

- [Geomorphology-based artificial neural networks (GANNs) for estimation of direct runoff over watersheds](https://doi.org/10.1016/S0022-1694(02)00313-X)

Dr. Shen 这边的Tad已经就马斯京根方法本身做了一些 physics-guided machine learning 的工作，基本思路应该来自 结合彭曼公式和ANN的那篇JGR的论文。具体的思路tad还没参与组会讲过，我还没看到。

### 人类活动对流域径流预测的影响-LSTM

本项目已移出至：https://github.com/OuyangWenyu/hydro-anthropogenic-lstm

### 使用更多遥感数据来帮助有水库流域的径流预测

在分析人类活动影响时，当想要单独分析水库的影响时，一个重要的事实是水库到底在哪，水库库容的变化怎样的等信息并不是那么容易获取的。

如果针对某些自己掌握完整调度运行资料的水库，可以通过对出流入流关系的拟合，提取调度规则，但是如果要进行大面积的分析，根据目前的信息情况，就需要采取不同的研究策略了。

在有一两个水库的流域上结合遥感数据分析水库对径流预测影响的论文可以参考：

- Hydrological model calibration for dammed basins using satellite altimetry information

其他关于水库和运行模拟的文献可以多参考杨大文老师那边一个学生做的两篇JH上机器学习和水库调度、径流预测结合的文章。

关于大范围内的分析，有一些不同的思路，尽可能利用更多遥感数据来获取流域预测的相关信息并融入建模中是一个思路。
比如现在已经有人做了根据MODIS数据来生成的500m分辨率2001年-2016年地表水面变化数据，这些数据可以想想如何纳入到现在的框架中，
还有SMAP数据，GRACE数据等，对不同数据，由于分辨率等问题，采用的策略必然会有所不同。这应该不是特别难，主要是选择一些比较好用的方法。

### 径流预测中初始状态的影响

目前似乎N-to-1的LSTM比N-to-N的更容易得到较好的效果，因为目前N-to-N的采用了比较好的dropout也没有得到更好的结果，
所以猜想N-to-1的有一个前期的warmup使得其更容易得到好结果，当然训练花费的时间也更多了。
目前可以确定在LSTM中，一定时间的warmup至少可以加快loss下降的速度，效果还需要进一步确定。

此外，根据宋天宇师兄的博士论文，可以期待在初始时刻同化一组径流来衡量warmup的效果。
如果warmup是有用的，那么我们可以期待在有水库的流域，可以尝试用它来进行初始状态估计，不过根据以往的效果来看，在没有更多信息进入的情况下，难发挥作用，不过可以分类看看其在大水库流域的效果。

另外，宋天宇师兄论文里面讨论的流域是南方的流域，对象是山洪，在这种条件下，确实前期状态是可以忽略的，而更广泛的应用中，初始态是重要的，所以可以考虑同化一些数据作为初始态，比如SMAP的数据。
这对于至少灌溉为主的流域，根据郝震的论文，应该是有较好的帮助的。

以及，在宋师兄的论文中，前期降雨时间序列的时间选择是按照平均汇流时间来选的，这里面就可以讨论初始启动相关的问题了，是不是平均汇流时间就是最好的？为什么不从降雨就开始？因为没办法可变序列？可变序列我们是不能补零的，
如果能够将warmup利用起来，我们可以将前面时刻的状态用warmup来补充。

具体来说，我可以首先做几件事，第一，判断下warmup是否有帮助，这部分可以在camels数据集上，或者现在3000+流域上做

第二，可以把模拟水库storage的那一套方法再拿来实验下。
可能可行的原因是warmup的信息来自历史径流，大鹏的DI(1)就已经能达到很好的效果了，而历史径流率定下的一个输出作为初始状态也是有可能达到一定的效果的，不过CNN kernel提取的结果确实没有表现出更好的性能，所以可能期待不用太多。
大鹏的论文里面，头一天的径流同化进来效果是最好的，时间太长的径流同化效果反而会差一点点，所以可以考虑选择一段较短时间的历史径流来推求kernel作为初始态。

## rough ideas

一些还没想太清楚大概怎么实施，但是直观上理解应该是能做些东西的想法，或者实施起来需要工作量较大的想法放在这里。

- 径流预测迁移学习
- FDC等水文签名与ML结合的相关问题
- 同化雷达测雨的小时尺度序列径流预测
- 灌溉信号分析预测

### PUB径流预测

在欧美发达国家，丰富的水文水资源观测网络使得他们有充足的数据开展相关研究，而在包括我国在内的广大发展中国家中，观测体系还没有达到同等水平，
或者数据资料不容易获取，在这种条件下可以考虑利用迁移学习来帮助径流预测。

不同流域之间的水文机理存在一定共性，有理由相信在数据观测充足的流域拟合得到的模型可以帮助我们认识缺资料地区的径流预测。
根据Kai Ma的论文，他将CAMELS数据集训练的模型迁移到国内受人类活动影响较小的流域，还有智利以及英国，
目前的结果说明从美国诸多流域中训练出的模型能启发其他地方流域，这说明了流域水文响应机理之间是有相通性的。

不过流域之间水文机理必然也有非共性，因此猜想不同迁移学习策略对不同流域的迁移效果会有所不同。
常见的策略比如将最后一层设置为未训练的，其他权重保持不变，看结果如何。

现在已经有了好几篇相关的论文了，有必要都阅读下，然后考虑几个问题：

（1）根据Kai的介绍，迁移学习和直接本地训练的相比，小流域能得到不错的结果，而大流域不能，可以猜想应是大流域的heterogeneous表现得比较强，而forcing data均化后无法体现这种异构性，
因此大流域表现比较差，可以通过将大流域拆分+流域回归河道汇流来探讨这一假说是否正确。而大流域拆分就可以考虑google的Hydronets有没有帮助。
（2）再比如东北流域/三峡流域，人类活动水利工程影响比较显著，和我目前做的东西也比较契合，可以考虑分析这方面的因素。
（3）另外，embedding 这类策略也是减少数据使用的方式之一，这在具体策略上对迁移学习也是有帮助的。
（4）最后就是迁移学习涉及的具体技术细节问题挺多的，比如当部分数据缺失时，如何迁移？还需要结合文献进一步思考。

关于PUB问题的思考可以参考以下PUB径流模拟预测相关文献：

- [A decade of Predictions in Ungauged Basins (PUB)—a review](https://doi.org/10.1080/02626667.2013.803183)
- [The value of regionalised information for hydrological modelling](https://core.ac.uk/download/pdf/76990578.pdf)
- [Flow Prediction in Ungauged Catchments Using Probabilistic Random Forests Regionalization and New Statistical Adequacy Tests](https://doi.org/10.1029/2018WR023254)
- [Calibration of the US Geological Survey National Hydrologic Model in Ungauged Basins Using Statistical At-Site Streamflow Simulations](https://ascelibrary.org/doi/full/10.1061/%28ASCE%29HE.1943-5584.0001854)

技术上关于LSTM 迁移学习方面的内容，有相似的文献可以参考，比如：

- [Deep Transfer Learning for Crop Yield Prediction with Remote Sensing Data](https://dl.acm.org/doi/abs/10.1145/3209811.3212707)
- [Transfer Learning from Deep Features for Remote Sensing and Poverty Mapping](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12196)
- [Combining satellite imagery and machine learning to predict poverty](https://science.sciencemag.org/content/353/6301/790)
    - 及其补充资料[Supplementary Materials for Combining satellite imagery and machine learning to predict poverty](www.sciencemag.org/content/353/6301/790/suppl/DC1)

Crop Yield Prediction这篇文章，通过遥感图像来提取信息的前提是图像中的信息本身得和要预测的内容之间有一定逻辑上的联系，
比如crop yield能用图像直接预测的原因是图像的颜色变化能够反映出植物的生长情况。所以如果想要通过图像获取一些关心的信息，首先要看看图像中是不是包含所需要的信息。

### FDC等水文签名与ML结合的相关问题

FDC不仅在水资源管理设计中占有重要地位，它还是水文预测中一种重要的水文签名，对PUB问题来说是一项重要的计算指标。

通过data driven获取FDC的方式常见的是对各个流域回归FDC中的各个quantile流量与流域attributes之间的关系，这样根据流域属性就可以获取quantile值了，然后对于测试流域就可以去预测FDC了。
这比较适合决策树来做，数据量不大，结果可解释性会比较强，能分析作为自变量的各类属性中，哪些是比较discriminant的。比如这篇论文：
[Prediction of regional streamflow frequency using model tree ensembles](http://dx.doi.org/10.1016/j.jhydrol.2014.05.029)

同样的思路，利用神经网络，构建多输入多输出的模型，可以考虑各个quantile之间的multicollinearity，即quantile之间的关系，单调性。能达到更好的效果。参考这篇文献：
[Prediction and Inference of Flow Duration Curves Using Multioutput Neural Networks](https://doi.org/10.1029/2018WR024463)

结合FDC的idea，大鹏已经做了一些相关工作，论文就要发了，他是分析FDC的加入对于径流预测能否用帮助，因为国内很多流域由于工程实践会做设计洪水分析，所以虽然没有径流资料，但是很多都有FDC。
所以对国内无观测流域是有用的。

所以我需要结合目前手头在做的工作，考虑一些新的点，首先因为国内很多FDC是由于有水利工程才有的，有了工程之后，变化是显著的，所以如何加入这部分的分析是需要进一步调研的。

另外，相比于直接做径流预测，做signature，一方面可以直接考虑人类活动影响属性，比较适合迁移回受人类活动影响很多的国内；另一方面，数据方面可能更好处理一些，可做性比较强。

并且，这块主要可以和三峡相关的雨洪相似分析的项目结合起来，可以再进一步思考一些更细致的想法。

可以参考的文献：The value of regionalised information for hydrological modelling的Chapter 7.

### 同化雷达测雨的小时尺度序列径流预测-seq2seq encoder-decoder LSTM

在实际洪水预报，以及防洪调度，发电调度等实际工作中，我们不仅仅关心日尺度的短期预测，也关心更精细时间尺度的径流预测，以便随时把握流域洪水动向。

因此，有必要在现有研究基础上构建一个sequence-to-sequence的LSTM模型，开展短时间尺度的洪水预报，为短期防洪提供数据支持。

seq2seq可以参考文献：[A Rainfall-Runoff Model with LSTM-based Sequence-to-Sequence Learning](https://doi.org/10.1029/2019WR025326) ，
这篇文章使用了不仅使用了历史数据，还同化了预报降雨数据。
我想是因为利用预报数据为seq2seq模型提供更多信息，以延长径流预测预见期是现阶段比较容易能想到的一种改善LSTM seq2seq不准问题的方法。

要在国内流域复现的话，可以尝试利用ECMWF，GFS，或者中国气象局等的降雨预报数据实验，可以先用观测降雨cheat一下，看看效果，然后再试着用预报数据做。

不过既然这篇文章已经做过了，就没有创新性了，怎么办？

首先就是可以尝试下日尺度的seq2seq计算，看看是否可行，但是估计不行。

所以另一方面，从国内具体问题出发，结合国内实际观测情况考虑如何在没有测雨站的地方，做同化雷达测雨的小时径流预测。

在我国境内，现已建成了较完备的多普勒雷达降雨观测及预报体系，比如在东北桓仁流域，雷达可以提供覆盖全流域的降雨数据，可以弥补地面雨量站观测不足的缺点。

另外，根据这篇文章：[Including spatial distribution in a data‐driven rainfall‐runoff model to improve reservoir inflow forecasting in Taiwan](https://doi.org/10.1002/hyp.9559) ，
我们可以看到，降雨空间分布的信息对径流预测的机器学习模型是有帮助的，这可能是雷达测雨对水库径流预测的另一个益处。

因此我们可以利用LSTM同化雷达观测数据以提高短期径流预测的精度。但是国内雷达测雨数据需要一些额外处理。

首先，雷达测雨的原始数据是回波图，如果是气象局自己处理的降雨数据，可能处理地比较简单，再加上使用上信息不全很可能有误差，所以可能第一步都需要额外的处理，
关于回波图和降雨的关系还需要进一步探讨。

还有，利用ConvLSTM我们还可以开展短期降雨径流预报，这方面可以参考文献：http://papers.nips.cc/paper/7145-deep-learning-for-precipitation-nowcasting-a-benchmark-and-a-new-model

相关代码可以参考：
https://github.com/sxjscience/HKO-7
https://github.com/Hzzone/Precipitation-Nowcasting

这样我们可以实时同化预报降雨数据以构造更长的实时洪水预测预见期，可以在有降雨观测的流域，专门用雷达预报降雨来做实验，这样避免只用雷达测雨时候的一些还不太清楚怎么做的数据处理。

同化预报数据的论文可以参考：
[Development and evaluation of a hydrologic data-assimilation scheme for short-range flow and inflow forecasts in a data-sparse high-latitude region using a distributed model and ensemble Kalman filterin](https://doi.org/10.1016/j.advwatres.2019.06.004)

最后可以考虑多源数据的利用方面，比如类似这篇文章：[Value of different precipitation data for flood prediction in an alpine catchment: A Bayesian approach](https://doi.org/10.1016/j.jhydrol.2016.06.031)

### Physics-guided ML 灌溉用水分析

分析有水库流域的径流预测时，能够看出有灌溉的大水库流域的预测是比较差的，说明目前的信息不足够提供气象驱动和径流之间的相关性，
所以需要融合更多信息，其中会涉及到灌溉用水分析。

我在做水库对径流预测的影响问题时候会对灌溉用水分析这个问题做简化处理，然后结合到物理规则里来展开径流预测，详见 https://github.com/OuyangWenyu/hydro-anthropogenic-lstm。

不过关于灌溉用水分析本身，还有更多的东西能做，因为室友做这个，所以可以和他多交流下，
我能想到的，比如可能能够结合 crop-water-demand 这类有一定物理意义的方程，在国内案例上做，可能能做出泛化能力更强的model。

## 比较简单的想法

这部分再记录一些比较原始的想法。

- 水库大坝识别
- Runoff数据统计降尺度
- 强化学习和调度的初步结合

### 水库面积、水位等信息：Mask-RCNN

大尺度上基于图像识别来做更多工作，近年来，在图像识别方面，人工智能体现了其强大能力，另一方面，近年来，也有一系列的关于地表水变化的文献和数据资料不断出现，比如 
https://www.nature.com/articles/nature20584 ，
https://www.nature.com/articles/s41597-020-0362-5 等

这为人工智能监督学习提供了算法和标记资料，因此可以尝试开展这方面的工作，以获取更多关于水库的属性及动态变化的信息。

这主要是可以往 water-energy-nexus那个方向上去结合，比如有不少研究根据遥感图像识别太阳能电池板。

单纯识别水库地倒是还没充分调研，不过类似的识别水体的文章并不少见，比如：

- [Water Body Extraction from Very High Spatial Resolution Remote Sensing Data Based on Fully Convolutional Networks](https://www.mdpi.com/2072-4292/11/10/1162)
- [Measuring River Wetted Width From Remotely Sensed Imagery at the Subpixel Scale With a Deep Convolutional Neural Network](https://doi.org/10.1029/2018WR024136)
- [Monitoring Ethiopian Wheat Fungus with Satellite Imagery and Deep Feature Learning Reid](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Pryzant_Monitoring_Ethiopian_Wheat_CVPR_2017_paper.html)
- [Mapping landslides from EO data using deep-learning methods](https://meetingorganizer.copernicus.org/EGU2020/EGU2020-11876.html)
- [Combining satellite imagery and machine learning to predict poverty](https://science.sciencemag.org/content/353/6301/790)
- [Deep Learning for Remote Sensing Data](https://ieeexplore.ieee.org/abstract/document/7486259)

这块彭老师那边的陈任飞在做一些图像识别的东西，张老师这边也有人做一些图像识别相关的内容，都可以交流下。

### 统计降尺度

这个是因为之前提到的变尺度水文模型而想到的相关内容，这部分内容也是被很多人感兴趣的。

目前能想到的，觉得可能会有用的，比如GLDAS数据在中国的应用，为了获取更高精度的LDAS数据，可以利用GLDAS和NLDAS之间的关系开展统计降尺度拟合，然后在中国生成有NLDAS精度的GLDAS数据。
这里主要想的是runoff数据。

得到的LDAS的数据的方式之一就是利用VIC model，这方面顾学志有在做，所以可以尝试利用VIC+DL做中国的分析，然后结合一些关于径流reanalysis数据的论文，看看能不能做相关的东西。

这对无观测流域径流预测或者大流域径流预测也许都能提供一些帮助。

需要进一步阅读文献，可以参考如下文献：

问题上主要参考：

- [Sequence-based statistical downscaling and its application to hydrologic simulations based on machine learning and big data](https://doi.org/10.1016/j.jhydrol.2020.124875)
- [Simulation of streamflow with statistically downscaled daily rainfall using a hybrid of wavelet and GAMLSS models](https://doi.org/10.1080/02626667.2019.1630742)
- [Downscaling SMAP Radiometer Soil Moisture Over the CONUS Using an Ensemble Learning Method Peyman](https://doi.org/10.1029/2018WR023354)
- [Downscaling of Rainfall Extremes From Satellite Observations](https://doi.org/10.1029/2018WR022950)
- [An in-situ data based model to downscale radiometric satellite soil moisture products in the Upper Hunter Region of NSW, Australia](https://doi.org/10.1016/j.jhydrol.2019.03.014)
- [Global downscaling of remotely sensed soil moisture using neural networks](https://ntrs.nasa.gov/search.jsp?R=20180006912)
- [Using Sensor Data to Dynamically Map Large-Scale Models to Site-Scale Forecasts: A Case Study Using the National Water Model](https://doi.org/10.1029/2017WR022498)

技术上主要参考：

- [Statistical downscaling of precipitation using long short-term memory recurrent neural networks](https://doi.org/10.1007/s00704-017-2307-2)
- [DeepSD: Generating High Resolution Climate Change Projections through Single Image Super-Resolution](https://doi.org/10.1145/3097983.3098004)
- [Improving Precipitation Estimation Using Convolutional Neural Network](https://doi.org/10.1029/2018WR024090)

### 强化学习与调度决策

作为一种序列决策的通用范式，强化学习展示出在调度方面的一些应用潜力。可以考虑通过问卷调查的形式获取一些强化学习所需的参数，
比如算法中的计算reward的discount可以通过问卷调查的形式获取，或者通过构建相关分析来获得，然后和现有的优化算法结合，去预测调度决策的结果。

这方面文章也有一些，教研室也有人做，可以多和他们交流交流。

## DL代码环境配置

建议安装本repo所需的库到一个服务器上，这样就可以通过远程到这台服务器上直接使用该库，以免重复安装和版本问题。

首先要在一个服务器上安装anaconda，并创建一个conda environment，命名为hydrodl，具体方式可以参考：
[OuyangWenyu/hydrus/1-basic-envir/2-python-envir.ipynb](https://github.com/OuyangWenyu/hydrus/blob/master/1-basic-envir/2-python-envir.ipynb)

执行以下语句安装lib：

```Shell
conda create --name hydrodl python=3.7
conda activate hydrodl
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -c conda-forge geopandas
# read excel in pandas
conda install -c conda-forge xlrd
conda install -c conda-forge geoplot
conda install -c conda-forge bayesian-optimization
conda install -c conda-forge tensorboard
conda install -c conda-forge kaggle
conda install -c conda-forge pydrive
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install skorch
conda env export > environment.yml
```

## 使用说明

推荐在本地机器上使用pycharm的ssh功能连接到此python执行环境。这样就能利用远程机器和数据进行计算了。

在实验阶段，请将应用程序写在test文件夹中，以unittest的形式开展测试，最后再将能形成合理计算结果的代码整理到app文件夹下，以供别人使用。

各个文件夹的具体作用参考下节。

## repo结构说明

各个文件夹的说明如下：

- app：该文件夹用于存放可以执行出合理计算结果的脚本代码；
- data：关于数据配置，读取和转换等数据前处理方面的代码；
- example：存放用于计算的数据；
- explore：统计计算分析相关程序；
- hydroDL：深度学习算法应用的核心文件夹；
- refine：超参数优化相关；
- test：测试文件；
- utils：常用的通用工具；
- visual：可视化相关。

definitions文件定义了项目执行的基本文件环境。

整个repo代码执行时基本的pipeline是这样的：

1. 数据获取和数据清理，及格式化为模型输入格式；
2. 探索性数据分析，包括统计计算和初步可视化，对数据分布等信息有初步掌握，补充第一个环节；
3. 核心环节，构建模型，拟合数据；
4. 比较不同超参数下机器学习模型的性能；
5. 执行超参数调优获取最优模型；
6. 最后测试集上评价最优模型并可视化。
