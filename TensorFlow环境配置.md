<h1 align = "center">TensorFlow环境配置🚀</h1>

<div style="text-align: right; color: Purple; font-size: 18px;">
    <span style="font-weight: bold; font-style: italic; text-decoration: underline;">
        <a href="https://github.com/kestlermai" style="color: Purple;">--by:maihuanzhuo</a>
    </span>
</div>

## 第一步---配置python环境（Anaconda）

### 1.点击下载并安装：[Anaconda](https://www.anaconda.com/)（个人习惯一般安装放在D盘）

### 2.配置环境:

- 打开系统属性，点击环境变量;（温馨提示：在`cmd+r`打开终端terminal，输入`sysdm.cpl`，在`高级`中打开`环境变量`）
- 在系统变量中找到`Path`变量，选择后点击编辑;
- 新建添加三个变量（根目录、Scripts目录、Library下bin目录）；

🙌示例：

```
D:\Anaconda
D:\Anaconda\Scripts
D:\Anaconda\Library\bin
```

- `win+r`输入`cmd`，然后在命令窗口输入`conda -V`命令，显示conda版本就代表配置成功；
- 打开`Anaconda Navigator`选择打开python IDE--spyder、pycharm、vscode任君选择；推荐在安装CUDA之前安装好vscode

---

## 第二步---Cuda和Cudnn的安装（安装CPU版本的，不是独立的NVIDIA显卡都可以忽略）

### 1.Cuda安装

- 查看CUDA版本：`WIN+X`打卡终端Windows PowerShell，然后输入`nvidia-smi`查看CUDA版本

```powershell
PS C:\Users\maihuanzhuo> nvidia-smi
Tue Mar 19 18:22:02 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 546.29                 Driver Version: 546.29       CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  | 00000000:01:00.0  On |                  N/A |
| N/A   50C    P5               8W / 122W |   1985MiB /  8188MiB |     22%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

- [这里可以看显卡对应的算力](https://developer.nvidia.com/cuda-gpus)，下载[CUDA](https://developer.nvidia.com/cuda-toolkit)对应电脑的版本，默认安装即可

- 安装完CUDA后，`WIN+X`打卡终端Windows PowerShell，然后输入`nvcc --version`，出现对应的CUDA版本信息

```powershell
PS C:\Users\maihuanzhuo> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Nov__3_17:51:05_Pacific_Daylight_Time_2023
Cuda compilation tools, release 12.3, V12.3.103
Build cuda_12.3.r12.3/compiler.33492891_0
```

### 2.CUDNN安装

- 下载地址：https://developer.nvidia.com/cudnn（鸡哥说他的4090系装的是CUDNN8.2）

- 注册英伟达账号，然后申请验证一下，通过后即可下载对应CUDA版本的CUDNN（CUDA12.2对应CUDNN8.9.3）

- 然后解压之后，除了==LICENSE==文件之外，其他三个文件夹复制到==C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2==目录下

- 接着添加CUDNN环境变量

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include
```

- 检查是否安装成功，`win+r`输入`cmd`打开终端

```cmd
cd C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\extras\demo_suite
```

- 分别运行 ==deviceQuery.exe==和==bandwidthTest.exe==，出现PASS则为成功

```powershell
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\extras\demo_suite>deviceQuery.exe
deviceQuery.exe Starting...
................................
deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.3, CUDA Runtime Version = 12.3, NumDevs = 1, Device0 = NVIDIA GeForce RTX 4060 Laptop GPU
Result = PASS

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\extras\demo_suite>bandwidthTest.exe
[CUDA Bandwidth Test] - Starting...
Running on...
................................
Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

- 查看CUDNN版本，在==C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include==目录下，打开==cudnn_version.h==文件

```cmd
C:\Users\maihuanzhuo>cd C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include>cudnn_version.h

#define CUDNN_MAJOR 8
#define CUDNN_MINOR 9
#define CUDNN_PATCHLEVEL 6
#因此我的cudnn版本是8.9.6
```

> 更多细节可参考：[详解 Windows 10 安装 CUDA 和 CUDNN_win10 cudnn-CSDN博客](https://blog.csdn.net/KRISNAT/article/details/130966344)

---

## 第三步---安装TensorFlow

### 创建tensorflow环境

- 首先根据[TensorFlow官网安装教程](https://tensorflow.google.cn/install/source_windows?hl=zh-cn#tested_build_configurations) ==tensorflow-2.6==要求==Python3.6-3.9==，CUDA跟CUDNN应该是往下兼容的

- 因为spyder默认安装的是python 3.10或者是3.11，因此我们要创建一个虚拟环境env（跟linux一样），在``终端terminal``中输入：

```cmd
conda create -n tensorflow_2.6 python=3.9 -y
```

- 创建环境后激活环境

```cmd
(base) C:\Users\maihuanzhuo>conda activate tensorflow_2.6

(tensorflow_2.6) C:\Users\maihuanzhuo>
```

### 通过conda或者pip安装TensorFlow

```cmd
conda install tensorflow-gpu==2.6.0 -y #GPU版本
conda install tensorflow==2.6.0 -y #CPU版本
pip install tensorflow-gpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple #GPU版本，网不行的可以加个镜像
pip install tensorflow==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple #CPU版本
```

### 安装成功后

- 在``terminal``中输入`python`打开python交互界面，导库检查是否报错`import tensorflow as tf`

```python
import tensorflow as tf
print(tf.__version__)
```

```cmd
(tensorflow_2.6) C:\Users\maihuanzhuo>python
Python 3.9.18 (main, Sep 11 2023, 13:30:38) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version__)
2.6.0
```

- 没有报错就代表安装成功，输入quit()来退出

---

## 在tensorflow环境下安装spyder

- 正常来说应该是在tensorflow_2.6环境下conda install spyder，不知道为什么一直装不上
- 在Anaconda Navigator页面里，spyder是显示没有安装的，所以比较麻烦
- 那么我们就通过pip源来安装spyder

```cmd
pip install spyder -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 通过pip源安装后，有个问题，现在进入spyder需要先激活tensorflow_2.6环境，然后在cmd prompt中输入spyder启动

---

## 安装tensorflow后发现一系列的依赖库报错解决

这个算是tensorflow的老毛病吧，迟迟没有更新，毕竟从1.0推出至今快十年了，当然现在更多是推荐pytorch。pytorch安装方法也一样，安装好了CUDA和CUDNN之后，为pytorch创建一个新环境，官方要求python版本为3.8-3.11，创建好之后激活环境，然后在[pytorch官网](https://pytorch.org/get-started/locally/)；根据对应配置复制代码，在新激活的环境中输入自动下载pytorch，同样也通过pip源安装spyder

请注意：根据CUDA版本对应安装，一般来说CUDA都会往下兼容，自身电脑CUDA版本太高也没关系

```powershell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 #CUDA 12.1
pip3 install torch torchvision torchaudio #CPU
```

当我们配置好env进入spyder，导库的时候会发现numpy报错，这是因为numpy版本太高，已经不支持 `np.object`所导致

```cmd
AttributeError: module 'numpy' has no attribute 'object'.
`np.object` was a deprecated alias for the builtin `object`. To avoid this error in existing code, use `object` by itself. Doing this will not modify any behavior and is safe.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```

解决：将numpy版本修复到仍然支持使用 `np.object` 的最后一个版本1.23.4

```cmd
pip uninstall numpy
pip install numpy==1.23.4
```

当然也会出现如下一系列库的版本不兼容，以及缺乏相应的库

```cmd
ERROR: ERROR: pip’s dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.6.0 requires clang=5.0, which is not installed.
tensorflow 2.6.0 requires keras>=2.4.0, which is not installed.
pandas 2.2.1 requires numpy<2,>=1.22.4;python version<“3.11”, but you have numpy 1.20.0 which is incompatible.
scipy 1.11.4 requires numpy<1.28.0, >=1.21.6, but you have numpy 1.20.0 which is incompatible.
tensorflow 2.6.0 requires google-auth<2, >=1.6.3, but you have google-auth 2.22.0 which is incompatible.
tensorflow 2.6.0 requires absl-py =0.10, but you have absl-py 1.4.0 which is incompatible.
tensorflow 2.6.0 requires flatbuffers =1.12, but you have flatbuffers 20210226132247 which is incompatible.
```

那就需要一一对应安装所提示的库，譬如需要requires keras>=2.4.0

```cmd
pip install keras==2.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

