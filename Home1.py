# 主要用于重构一些函数
import os
import numpy as np
import pywt
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import periodogram
from scipy.signal import lfilter,stft


fs=200000
time=np.arange(0,10,1/200000)
time=time[::50]
# time=np.arange(0,10,1/80)
# 频域分析,计算轴承的故障频率，轴承型号为ER16K
fr=0
D=38.52
d=7.94
num_ball=9
BPFI=5.43*fr #内圈故障频率
BPFO=3.57*fr #外圈故障频率
BPFB=4.64*fr

# 使用列表传参叭
def draw_four_pic(ch,figsize=(12, 8),hspace=0.3,color='blue',save=None):
    plt.figure(figsize=figsize)  # 设置图形的大小
    plt.subplots_adjust(hspace=hspace)  # 调整水平间距
    # 1
    plt.subplot(221)  # 4行1列，第1个子图
    plt.plot(time, ch[0], color=color)
    plt.title('HA1-Amplitude-Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线
    # 2
    plt.subplot(222)  # 4行1列，第1个子图
    plt.plot(time, ch[1], color=color)
    plt.title('IA1-Amplitude-Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线

    # 3
    plt.subplot(223)  # 4行1列，第1个子图
    plt.plot(time, ch[2], color=color)
    plt.title('OA1-Amplitude-Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线

    # 4
    plt.subplot(224)  # 4行1列，第1个子图
    plt.plot(time, ch[3], color=color)
    plt.title('BA1-Amplitude-Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线

    if save!=None:
        plt.savefig(f"第一次作业图像/{save}.png",dpi=180)
    plt.show()  # 显示图形


# 设置滤波器
def filter(ch,Wn=[5000,99000],order=4,btype='band',fs=fs):
    order=5 #滤波的介词
     #截止频率 
    filter_ch=[]
    for c in ch:
        b, a = signal.butter(order, Wn=Wn, btype=btype, analog=False, output='ba',fs=fs)
        filter_c= signal.lfilter(b, a, c)
        filter_ch.append(filter_c)
    return filter_ch

def filt(ch,Wn=[5000,99000],order=4,btype='band',fs=fs):
    b, a = signal.butter(order, Wn=Wn, btype=btype, analog=False, output='ba',fs=fs)
    filter_c= signal.lfilter(b, a, ch)
    return filter_c

# 设置去均值
def del_mean(ch):
    mean_ch=[]
    for c in ch:
        mean_value=np.mean(c)
        mean_c=c-mean_value
        mean_ch.append(mean_c)
    return mean_ch

def del_m(ch):
    mean_value=np.mean(ch)
    mean_c=ch-mean_value
    return mean_c

# 归一化函数
def norm(ch):
    norm_ch=[]
    for c in ch:
        max_value=np.max(c)
        min_value=np.min(c)
        norm_c=(c-min_value)/(max_value-min_value)
        norm_ch.append(norm_c)
    return norm_ch

def nor(ch):
    max_value=np.max(ch)
    min_value=np.min(ch)
    norm_c=(ch-min_value)/(max_value-min_value)
    return norm_c

# 计算有量纲参数，更新版本
def cal_none1(ch_list,slove=None):
    # 存储四种不同故障类型
    sqrt_type=[] #方根幅值
    aver_type=[] #平均幅值
    square_type=[] #均方幅值
    # 一组信号
    for chh in ch_list:
        sqrt_speed=[] # 单一故障类型的有量纲指标，12组
        aver_speed=[]
        square_speed=[]
        for ch1 in chh:
            if slove=='filter':
                ch1=filt(ch1)
            elif slove=='mean':
                ch1=del_m(ch1)
            elif slove=='norm':
                ch1=nor(ch1)
            sqrt_ch= (np.average(np.sqrt(np.abs(ch1))))**2
            aver_ch=np.average(ch1)
            square_ch=np.sqrt(np.average(np.square(ch1)))
            sqrt_speed.append(sqrt_ch)
            aver_speed.append(aver_ch)
            square_speed.append(square_ch)
        sqrt_type.append(sqrt_speed)
        aver_type.append(aver_speed)
        square_type.append(square_speed)

    sqrt_type=np.array(sqrt_type)
    aver_type=np.array(aver_type)
    square_type=np.array(square_type)
    return sqrt_type,aver_type,square_type

# 无量纲参数计算
def cal_none(folder_path,slove=None):
    # 存储四种不同故障类型
    sqrt_type=[] #方根幅值
    aver_type=[] #平均幅值
    square_type=[] #均方幅值
    for folder in folder_path:
        sqrt_speed=[] # 单一故障类型的有量纲指标，12组
        aver_speed=[]
        square_speed=[]
        for item in os.listdir(folder):
            item_path=os.path.join(folder,item)
            data=loadmat(item_path)
            ch1=data['Channel_1']
            if slove=='filter':
                ch1=filt(ch1)
            elif slove=='mean':
                ch1=del_m(ch1)
            elif slove=='norm':
                ch1=nor(ch1)
            sqrt_ch= (np.average(np.sqrt(np.abs(ch1))))**2
            aver_ch=np.average(ch1)
            square_ch=np.sqrt(np.average(np.square(ch1)))
            sqrt_speed.append(sqrt_ch)
            aver_speed.append(aver_ch)
            square_speed.append(square_ch)
        sqrt_type.append(sqrt_speed)
        aver_type.append(aver_speed)
        square_type.append(square_speed)

    sqrt_type=np.array(sqrt_type)
    aver_type=np.array(aver_type)
    square_type=np.array(square_type)
    return sqrt_type,aver_type,square_type


def draw_none_4(sqr,ave,squ,figsize=(12, 8),hspace=0.3,save=None):
    plt.figure(figsize=figsize)  # 设置图形的大小
    plt.subplots_adjust(hspace=hspace)  # 调整水平间距
    sample=np.arange(50)
    # 1
    plt.subplot(221)  # 4行1列，第1个子图
    plt.plot(sample, sqr[0], color='blue', marker='o', label='Square Root (sqt)')
    plt.plot(sample, ave[0], color='green', marker='s', label='Average (aver)')
    plt.plot(sample, squ[0], color='red', marker='^', label='Square (square)')
    # plt.plot(sample, sqr[0], color='blue',  label='Square Root (sqt)')
    # plt.plot(sample, ave[0], color='green',  label='Average (aver)')
    # plt.plot(sample, squ[0], color='red',  label='Square (square)')
    plt.title('HA-Sample')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线
    plt.legend()  # 添加图例

    # 2
    plt.subplot(222)  # 4行1列，第2个子图
    plt.plot(sample, sqr[1], color='blue', marker='o', label='Square Root (sqt)')
    plt.plot(sample, ave[1], color='green', marker='s', label='Average (aver)')
    plt.plot(sample, squ[1], color='red', marker='^', label='Square (square)')
    plt.title('IA-Sample')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线
    plt.legend()  # 添加图例

    # 3
    plt.subplot(223)  # 4行1列，第3个子图
    plt.plot(sample, sqr[2], color='blue', marker='o', label='Square Root (sqt)')
    plt.plot(sample, ave[2], color='green', marker='s', label='Average (aver)')
    plt.plot(sample, squ[2], color='red', marker='^', label='Square (square)')
    plt.title('OA-Sample')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线
    plt.legend()  # 添加图例

    # 4
    plt.subplot(224)  # 4行1列，第4个子图
    plt.plot(sample, sqr[3], color='blue', marker='o', label='Square Root (sqt)')
    plt.plot(sample, ave[3], color='green', marker='s', label='Average (aver)')
    plt.plot(sample, squ[3], color='red', marker='^', label='Square (square)')
    plt.title('BA-Sample')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线
    plt.legend()  # 添加图例
    if save!=None:
        plt.savefig(f"第一次作业图像/{save}.png", dpi=180)
    plt.show()


# 波形分析
# 波形指标
def W(data):
    rms = np.sqrt(np.mean(data**2))
    mean_data=np.mean(data)
    return rms/mean_data

# 峰值指标
def C(data):
    rms = np.sqrt(np.mean(data**2))
    peak_value = np.max(np.abs(data))
    return peak_value/rms

# 脉冲指标
def I(data):
    peak_value = np.max(np.abs(data))
    mean_data=np.mean(data)
    return peak_value/mean_data

# 裕度指标
def L(data):
    peak_value = np.max(np.abs(data))
    crest_factor = peak_value / np.sqrt(np.mean(data**2))
    return crest_factor

#  偏斜度指标
def S(data):
    return skew(data)

# 峭度指标
def K(data):
    return kurtosis(data)

def cal_have(ch_list,slove=None):
    Ws,Cs,Is,Ls,Ss,Ks=[],[],[],[],[],[]
    for chh in ch_list:
        w,c,i,l,s,k=[],[],[],[],[],[]
        for ch1 in chh:
            # item_path=os.path.join(folder,item)
            # data=loadmat(item_path)
            # ch1=data['Channel_1']
            if slove=='filter':
                ch1=filt(ch1)
            elif slove=='mean':
                ch1=del_m(ch1)
            elif slove=='norm':
                ch1=nor(ch1)
            w1=W(ch1)
            c1=C(ch1)
            i1=I(ch1)
            l1=L(ch1)
            s1=S(ch1)
            k1=K(ch1)
            w.append(w1)
            c.append(c1)
            i.append(i1)
            l.append(l1)
            s.append(s1)
            k.append(k1)
        Ws.append(w)
        Cs.append(c)
        Is.append(i)
        Ls.append(l)
        Ss.append(s)
        Ks.append(k)

    Ws=np.array(Ws)
    Cs=np.array(Cs)
    Is=np.array(Is)
    Ls=np.array(Ls)
    Ss=np.array(Ss)
    Ks=np.array(Ks)
    return [Ws,Cs,Is,Ls,Ss,Ks]

def draw_have_6(index,save=None):
    Ws,Cs,Is,Ls,Ss,Ks=index
    plt.figure(figsize=(12, 8))  # 设置图形的大小
    plt.subplots_adjust(hspace=0.3)  # 调整水平间距
    sample=np.arange(50)
    # 1
    plt.subplot(221)  # 4行1列，第1个子图
    plt.plot(sample, Ws[0], color='blue',marker='o',label='W')
    plt.plot(sample, Cs[0], color='green',marker='s',label='C')
    plt.plot(sample, Is[0], color='red',marker='^',label='I')
    plt.plot(sample, Ls[0], color='blue',marker='*',label='L')
    plt.plot(sample, Ss[0], color='pink',marker='+',label='S')
    plt.plot(sample, Ks[0], color='black',marker='.',label='K')

    plt.title('HA-Sample')
    plt.xlabel('Sample')
    plt.ylabel('Index')
    plt.grid(True)  # 添加网格线
    plt.legend()

    # 2
    plt.subplot(222)  # 4行1列，第1个子图
    plt.plot(sample, Ws[1], color='blue',marker='o',label='W')
    plt.plot(sample, Cs[1], color='green',marker='s',label='C')
    plt.plot(sample, Is[1], color='red',marker='^',label='I')
    plt.plot(sample, Ls[1], color='blue',marker='*',label='L')
    plt.plot(sample, Ss[1], color='pink',marker='+',label='S')
    plt.plot(sample, Ks[1], color='black',marker='.',label='K')
    plt.title('IA-Sample')
    plt.xlabel('Sample')
    plt.ylabel('Index')
    plt.grid(True)  # 添加网格线
    plt.legend()

    # 3
    plt.subplot(223)  # 4行1列，第1个子图
    plt.plot(sample, Ws[2], color='blue',marker='o',label='W')
    plt.plot(sample, Cs[2], color='green',marker='s',label='C')
    plt.plot(sample, Is[2], color='red',marker='^',label='I')
    plt.plot(sample, Ls[2], color='blue',marker='*',label='L')
    plt.plot(sample, Ss[2], color='pink',marker='+',label='S')
    plt.plot(sample, Ks[2], color='black',marker='.',label='K')
    plt.title('OA-Sample')
    plt.xlabel('Sample')
    plt.ylabel('Index')
    plt.grid(True)  # 添加网格线
    plt.legend()

    # 4
    plt.subplot(224)  # 4行1列，第1个子图
    plt.plot(sample, Ws[3], color='blue',marker='o',label='W')
    plt.plot(sample, Cs[3], color='green',marker='s',label='C')
    plt.plot(sample, Is[3], color='red',marker='^',label='I')
    plt.plot(sample, Ls[3], color='blue',marker='*',label='L')
    plt.plot(sample, Ss[3], color='pink',marker='+',label='S')
    plt.plot(sample, Ks[3], color='black',marker='.',label='K')
    plt.title('BA-Sample')
    plt.xlabel('Sample')
    plt.ylabel('Index')
    plt.grid(True)  # 添加网格线
    plt.legend()
    if save!=None:
        plt.savefig(f"第一次作业图像/{save}.png",dpi=180)
    plt.show()  # 显示图形

# 傅里叶变换
def DWT(ch,fs,num_samples=40000):
    fft_ch=[]
    for c in ch:
        f = np.linspace(0.0, (fs/2.0), num_samples//2)
        freq_values=fft(c)
        freq_values = 2.0/num_samples * np.abs(freq_values[0:num_samples//2])
        fft_ch.append([f,freq_values])

    return fft_ch

def period(ch):
    period_c=[]
    for c in ch:
        c=np.reshape(c,20000)
        fre, p_c = periodogram(c, fs=2000)
        period_c.append(p_c)
        print(len(fre),len(p_c))
    return period_c,fre

def llfilter(ch):
    ce_c=[]
    for c in ch:
        c=np.reshape(c,20000)
        ar_coeffs = [1.0, -0.9] #定义AR模型数
        ce = lfilter(ar_coeffs, 1, np.log(np.abs(np.fft.fft(c))))
        ce_c.append(ce)
    return ce_c


def SDWT(ch):
    sd_c=[]
    fres=[]
    Zxxs=[]
    for c in ch:
        c=np.reshape(c,20000)
        fre, time, Zxx = stft(c, fs=fs, nperseg=100)
        sd_c.append(time)
        fres.append(fre)
        Zxxs.append(Zxx)
    return sd_c,fres,Zxxs


def draw_8(ch1,ch2,fre=None,Zxx=None,save=None,title='FFT',btype="fft"):
    # 原始信号 傅里叶变换
    plt.figure(figsize=(12, 8))  # 设置图形的大小
    plt.subplots_adjust(hspace=0.8)  # 调整水平间距
    # 1
    plt.subplot(421)  # 4行1列，第1个子图
    plt.plot(time, ch1[0], color='blue')
    plt.title('HA1-Amplitude-Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线

    # 2
    plt.subplot(422)  # 4行1列，第1个子图
    if btype=='fft' or btype=='lfilter':
        plt.plot(ch2[0][0],ch2[0][1])
    elif btype=='periodogram':#自功率谱
        plt.semilogy(fre, ch2[0])
    elif btype=='stft':
        plt.pcolormesh(ch2[0], fre[0], 10 * np.log10(np.abs(Zxx[0])))
        plt.title('Short-Time Fourier Transform (STFT)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
    if btype!='stft':
        plt.title(f'HA1-{title}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线

    # 3
    plt.subplot(423)  # 4行1列，第1个子图
    plt.plot(time, ch1[1], color='blue')
    plt.title('IA1-Amplitude-Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线
    # 4
    plt.subplot(424)  # 4行1列，第1个子图
    if btype=='fft' or btype=='lfilter':
        plt.plot(ch2[1][0],ch2[1][1])
    elif btype=='periodogram':#自功率谱
        plt.semilogy(fre, ch2[1])
    elif btype=='stft':
        plt.pcolormesh(ch2[1], fre[1], 10 * np.log10(np.abs(Zxx[1])))
        plt.title('Short-Time Fourier Transform (STFT)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
    if btype!='stft':
        plt.title(f'HA1-{title}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线

    # 5
    plt.subplot(425)  # 4行1列，第1个子图
    plt.plot(time, ch1[2], color='blue')
    plt.title('OA1-Amplitude-Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线

    # 6
    plt.subplot(426)  # 4行1列，第1个子图
    if btype=='fft' or btype=='lfilter':
        plt.plot(ch2[2][0],ch2[2][1])
    elif btype=='periodogram':#自功率谱
        plt.semilogy(fre, ch2[2])
    elif btype=='stft':
        plt.pcolormesh(ch2[2], fre[2], 10 * np.log10(np.abs(Zxx[2])))
        plt.title('Short-Time Fourier Transform (STFT)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
    if btype!='stft':
        plt.title(f'HA1-{title}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线

    # 7
    plt.subplot(427)  # 4行1列，第1个子图
    plt.plot(time, ch1[3], color='blue')
    plt.title('BA1-Amplitude-Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线

    # 8
    plt.subplot(428)  # 4行1列，第1个子图
    if btype=='fft' or btype=='lfilter':
        plt.plot(ch2[3][0],ch2[3][1])
    elif btype=='periodogram':#自功率谱
        plt.semilogy(fre, ch2[3])
    elif btype=='stft':
        plt.pcolormesh(ch2[3], fre[3], 10 * np.log10(np.abs(Zxx[3])))
        plt.title('Short-Time Fourier Transform (STFT)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
    if btype!='stft':
        plt.title(f'HA1-{title}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
    plt.grid(True)  # 添加网格线
    if save!=None:
        plt.savefig(f"第一次作业图像/{save}.png",dpi=180)
    plt.show()  # 显示图形


def xiaobo(ch,wavelet = 'db4',save=None):
    # 进行小波变换
    level=4
    wavelet =wavelet  # 选择小波基函数
    coeffs = pywt.wavedec(ch, wavelet, level=level)  # 执行小波分解

    # 绘制小波系数
    plt.figure(figsize=(12, 8))
    for i, coeff in enumerate(coeffs):
        plt.subplot(len(coeffs), 1, i + 1)
        # reconstructed_signal = coeff.reconstruct(update = False) #从小波包系数重建信号
        # f, c = DWT(reconstructed_signal, fs, len(reconstructed_signal))
        # z = abs(c)
        # plt.plot(f,z)
        plt.plot(coeff)
        plt.title(f'Detail {i}' if i > 0 else 'Approximation')

    plt.tight_layout()



    if save!=None:
        plt.savefig(f'第一次作业图像/{save}.png')
    plt.show()

def apply_fft(x, fs, num_samples):
    f = np.linspace(0.0, (fs/2.0), num_samples//2)
    freq_values = fft(x)
    freq_values = 2.0/num_samples * np.abs(freq_values[0:num_samples//2])
    return f, freq_values

m=5
def xiaobo1(ch,wavelet ='db4',level=7,save=None):
    # 进行小波变换
    level=level
    wavelet =wavelet  # 选择小波基函数
    coeffs = pywt.WaveletPacket(ch, wavelet,maxlevel=level)  # 执行小波分解
    rewp=coeffs.reconstruct(update=False)
    
    f,c = apply_fft(rewp, fs, len(rewp))
    z = abs(c)
    plt.plot(f,z)
    z=np.reshape(z,20000)
    if save!=None:
        plt.savefig(f'第一次作业图像/{save}.png')
    plt.show()
    plt.tight_layout()
