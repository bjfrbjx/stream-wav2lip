# stream-wav2lip
参考仓库: https://github.com/Rudrabha/Wav2Lip ，https://github.com/primepake/wav2lip_288x288
  
以工程的思想，优化实现步骤，将头脸分离、嘴型替换、回补背景三个步骤分离。
在此基础上提前取脸，流式循环播放，对接obs。
因为原版的清晰度不够，遂找到288版调试，但288没有现成的预训练权重文件，只能自己搞。
LWR-1000数据集是大型中文唇语数据集，适合中文发音嘴型，但是搞这个要填邮件。
网上找半天，找到[云盘资料](https://blog.csdn.net/weixin_47907053/article/details/132039297),又是处理后的下半脸图片，勉强可以训练嘴型同步SyncNet网络，但后面wav2lip的训练就要自己去录视频了。
