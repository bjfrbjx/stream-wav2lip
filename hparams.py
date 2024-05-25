## audio
sample_rate = 16000
fmax = 7600
fmin = 55
num_mels = 80
n_fft=800
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
win_size = 800
hop_size = 200
max_abs_value = 4.

## define net
fps = 25
# 基于0.2秒-》fps=25->【5帧画面】，
syncnet_T = 5
# 基于0.2秒-》mel_num=80->【16】，
syncnet_mel_step_size = 16

## train
syncnet_eval_interval = 10000
img_size = 96
syncnet_lr = 1e-4
syncnet_batch_size = 192
eval_interval = 3000
batch_size = 16
initial_learning_rate = 1e-4
syncnet_wt = 0.0
disc_wt = 0.07
disc_initial_learning_rate = 1e-4