
# 需要进行时域、频域、时频域分析
import os
import numba
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import periodogram
from scipy.signal import lfilter
from Home1 import *



# 读取数据
HA1=loadmat(r'1-a healthy bearing\H-A-1.mat')
IA1=loadmat(r'2-a bearing with inner race fault\I-A-1.mat')
OA1=loadmat(r'3-a bearing with outer race fault\O-A-1.mat')
BA1=loadmat(r'4-a bearing with ball fault\B-A-1.mat')

# 分别读取两个通道的数值 ch1表示振动信号，ch2表示转速信号
# 采样频率为200KHz，采样时间为10s，分析振幅信号 分解成50组数据
ch_ha1=HA1['Channel_1']
ch_ia1=IA1['Channel_1']
ch_oa1=OA1['Channel_1']
ch_ba1=BA1['Channel_1']
# ch=[ch_ha1,ch_ia1,ch_oa1,ch_ba1]
# 40000个点一组数据
ch=[ch_ha1[::50],ch_ia1[::50],ch_oa1[::50],ch_ba1[::50]]

ch1=[ch_ha1[i::50] for i in range(50)]
ch2=[ch_ia1[i::50] for i in range(50)]
ch3=[ch_oa1[i::50] for i in range(50)]
ch4=[ch_ba1[i::50] for i in range(50)]
ch_50=[ch1,ch2,ch3,ch4]


# 绘制原始图像,分别是健康、内圈故障、外圈故障、球故障
draw_four_pic(ch=ch,save='原始数据')

# 数据预处理，进行滤波，这里使用高通滤波
filt_ch=filter(ch,Wn=90000,btype='low')
draw_four_pic(filt_ch,save='低通滤波')

# 归一化处理
norm_ch=norm(ch)
draw_four_pic(norm_ch,save='归一化处理')

# 时域有量纲分析
sqr,ave,squ=cal_none1(ch_list=ch_50)
draw_none_4(sqr,ave,squ,save='有量纲分析')

# 时域无量纲分析
index=cal_have(ch_list=ch_50)
draw_have_6(index,save='无量纲分析')

# 进行傅里叶变换
fft_ch=DWT(ch,fs=200000)
draw_8(ch,fft_ch,save="傅里叶变换")

# # 小波分解
high_fre=xiaobo1(ch_ia1[::50],wavelet='db4',save='正常信号小波分解')
print(high_fre)
