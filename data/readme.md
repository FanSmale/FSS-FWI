# 需知
本文件包含三个主要文件夹: CurveFaultA, CureveVelA, FlatFaultA.
这三个数据集均来自OpenFWI (数据下载地址: https://smileunc.github.io/projects/openfwi/datasets)
除开CurveVelA, 每个文件夹内都有108个速度模型npy文件.
其中每个npy文件内都有500个速度模型.

目前已经提供了所有数据的训练和测试速度模型(vmodelX.npy), 但是地震记录(seismicX.npy), 速度模型轮廓(cvmodelX.npy), 低分辨率重建速度模型(lvmodelX.npy)需要用户生成

1.地震记录是任何网络训练的基础.
它将通过根目录下的"forward2openfwi.py"来正演生成 (注意! 这个过程非常耗时).

2.速度模型轮廓是DDNet70和DenseInvNet训练的基础.
它将通过当前目录下的"create_contour_vmodel.py"来生成.

3.低分辨率重建速度模型是DenseInvNet训练的关键.
它需要用户通过自己训练的LResInvNet网络或者我们在models文件夹中已经提供好的网络来反演生成.
具体来说, 需要用户在test文件中选择命令: "-n {} -net LResInvNet -bv -nep .\models\{}Model\LResInvNet_100of100.pkl".
同时设置multi_or_single = 0, 并执行方法temp_tester.save_lowres_inversion_results()
通过这种方式, LResInvNet输出的速度模型会自动保存data文件夹的对应数据集内

最后, configuration文件夹内保存着不同数据集的训练和测试的分配策略.
"_base"表示InversionNet和VelocityGAN的配置文件.
"_cont"表示DDNet70的配置文件
"_lres"表示LResInvNet和DenseInvNet的配置文件.




