# preprocessing configs
in_res = (128, 128, 128)
out_res = (16, 16, 16)
win_min, win_max = -1000, 400
input_keys = ["airway", "artery", "vein", "lobe"]
resample_configs = {
    "airway": {"size": in_res, "interp": "nearest"},
    "artery": {"size": in_res, "interp": "nearest"},
    "vein": {"size": in_res, "interp": "nearest"},
    "lobe": {"size": in_res, "interp": "nearest"}
}

# model configs
enc_cfgs = {
    "in_channels": 9
}
dec_cfgs = {
    "num_channels": [963, 64],
    "num_classes": 3,
    "num_layers": 1,
    "drop_prob": 0.3
}
cor_cfgs = {
    "num_channels": [963, 64],
    "num_classes": 19,
    "num_layers": 1,
    "drop_prob": 0.3
}

gen_cfgs = {
    "num_channels": [1024, 1024, 512, 256, 128, 64, 32],
    "num_layers": 1,
    "num_classes": 19,
    'latent_dim': 1024
}


#dataset config
latent_dim = 1024

# training configs
w_ce = 0.5
w_dice = 1
w_def = 0.001
batch_size = 4
num_workers = 4
max_lr = 1e-3
min_lr = 1e-6
epochs = 100

# evaluation configs
eval_freq = 5
eval_res = (128, 128, 128)
eval_batch_size = 1

# inference configs
model_weights_path = "./models/model_weights.pth"
template_weights_path = "./models/pretrained_template_weights.pth"
infer_batch_size = 1
window_size = (96, 96, 96)
