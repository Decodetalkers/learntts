import torch
from utils import fix_len_compatibility

device = torch.device("cuda")
seed = 37
n_epochs = 1000
log_dir = "logs/new_exp"
log_dir2 = "logs/denoise"
learning_rate = 1e-5
batch_size = 10
batch_size2 = 20
denoise_batch_size = 4

f_min = 0
f_max = 8000
n_mels = 160
n_fft = 1024
sample_rate = 22050

out_size = fix_len_compatibility(2 * 22050 // 256)
