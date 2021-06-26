# hydro-routing-cnn
 ## 河道汇流-CNN

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
