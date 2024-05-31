# stream-wav2lip
参考仓库: https://github.com/Rudrabha/Wav2Lip ，https://github.com/primepake/wav2lip_288x288
  
以工程的思想，优化实现步骤，将头脸分离、嘴型替换、回补背景三个步骤分离。
在此基础上提前取脸，流式循环播放，对接obs。
因为原版的清晰度不够，遂找到288版调试，但288没有现成的预训练权重文件，只能自己搞。
LWR-1000数据集是大型中文唇语数据集，适合中文发音嘴型，但是搞这个要填邮件。
网上找半天，找到[云盘资料](https://blog.csdn.net/weixin_47907053/article/details/132039297),又是处理后的下半脸图片，勉强可以训练嘴型同步SyncNet网络，但后面wav2lip的训练就要自己去录视频了。


## 使用步骤

### 0. 处理自定义数据
有现成的数据集可以跳过这一级。  
准备一个文件夹，直接将所有带脸视频塞进去。 注意：  
1.所有视频都要25fps（模型里输入定死5帧图像和0.2秒音频对应，改了影响输出唇形判断结果）。
2.视频检查一下，不要有脸被挡、看不见嘴唇的片段，如果有会导致无法收敛
3.最好是正脸、略微侧脸的角度都有。
4.从头训练的话最好有各种各样的人脸，只有两三个模特的泛化能力弱。

使用 `python pre_detface4stream.py`直接批量预处理。默认使用cuda，没有显卡就降级cpu。
记得修改main中参数（懒得写命令行工具）
- face_dir: 视频文件夹
- nosmooth: 脸部框选地平滑一些（没什么用）
- out_dir: 导出文件夹（obs预处理数据）
- train_dir: 导出类似LSR结构的数据集文件夹
- face_size: 导出帧的高宽(尽量和后面训练模型的imgSize相同，原版是96,清晰的有288、384)

成功后会在out_dir下生成每个视频的obs预处理文件夹，train_dir下生成每个视频的训练用文件夹。
obs预处理文件夹下，是【bg,body_face,heads,head_face,audio.wav,face_det.npz】
- bg文件夹存放完整帧 
- body_face存放从身体中抠出的脸，是wav2lip采用的方法，直接用于wav2lip。
- heads是从身体上扣下的头，并调整角度，用于gfpgan增强
- head_face存放从heads扣出脸，用于wav2lip。
- audio.wav 16000采样率单声道的音频
- face_det.npz 存放将脸、头一一对应放回背景bg的定位角度信息

### 1. 导出视频
准备好训练好的权重文件，obs预处理文件夹，人语音频，outfile指定为希望导出的mp4文件。  
`python stream_out.py --checkpoint_path wav2lip288_step10000.pth --data_dir out_dir --audio_files  xxx.mp3 --outfile xx.mp4 `
因为原版的只导出权重没有网络结构，只能先根据权重文件的层数，写死对应的网络结构，看后面能不能优化。


### 2. 对接obs
准备好训练好的权重文件，obs预处理文件夹，人语音频，outfile指定为”obs“。  
打开obs，调好麦克风和画布比例。  
`python stream_out.py --checkpoint_path wav2lip288_step10000.pth --data_dir out_dir --audio_files  xxx.mp3 --outfile obs `


## 训练步骤

### 1. 训练Sycnet
在上一步的训练数据集文件夹里，添加train.txt 和 val.txt，分别写入训练数据和验证数据的全路径。

原版96x96的训练：`python ./Rudrabha_wav2lip/color_syncnet_train.py --data_root /root/data_dir --checkpoint_dir ./Rudrabha_wav2lip/checkpoint --pre_path ./Rudrabha_wav2lip/checkpoint/syncnet_step001751230.pth`

288的训练：`python ./primepake_wav2lip/x288/syncnet_train288.py --data_root /root/data_dir --checkpoint_dir ./primepake_wav2lip/checkpoint`

384的训练：`python ./primepake_wav2lip/x384/syncnet_train384.py --data_root /root/data_dir --checkpoint_dir ./primepake_wav2lip/checkpoint`

data_root就可以直接使用上一步的train_dir。用pre_path指定预训练文件

### 训练wav2lip
TODO： 暂时没有多人的高清视频，也没弄懂SAM和WGAN-GP在训练中的作用，后面再补。