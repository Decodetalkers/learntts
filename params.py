import torch

device = torch.device("cuda")
seed = 37
n_epochs = 50
log_dir = "logs/new_exp"
learning_rate = 1e-4
batch_size = 10

f_min = 0
f_max = 8000
n_mels = 80
n_fft = 1024
sample_rate = 16000
