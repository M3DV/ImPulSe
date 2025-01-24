import os
import random
from argparse import ArgumentParser
from importlib import import_module

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import SimpleITK as sitk
from tqdm import tqdm


from data.dataset import LungSegmentDataset
from data import transforms as aug
from models.tim import ImPulSe, ImPulSeDecoder
from models.resnet18 import ResNet3d18Backbone
from models.tim import TIm
from models.tim import TemplateGenerator
from utils.logger import logger


def _parse_cmd_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--gpu", default="0")
    arg_parser.add_argument("--cfg", default="lbav")
    args = arg_parser.parse_args()

    return args


def _set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_dataloaders():
    data_dir = "/lung3d"
    df = pd.read_csv("/lung3d.csv")

    transforms = [
        aug.GetLungBbox(),
        aug.CropLung(),
        aug.GetLobe(),
        aug.Resample(cfg.resample_configs),
        aug.OnehotEncode("lobe", 6),
        aug.SampleGrid(mode="uniform"),
        aug.ConcatInputs(cfg.input_keys),
        aug.ToTensor()
    ]

    ds_val = LungSegmentDataset(df, data_dir, transforms, "val")
    dl_val = LungSegmentDataset.get_dataloader(ds_val, cfg.infer_batch_size,
        False, cfg.num_workers, "infer")
    ds_test = LungSegmentDataset(df, data_dir, transforms, "test")
    dl_test = LungSegmentDataset.get_dataloader(ds_test, cfg.infer_batch_size,
        False, cfg.num_workers, "infer")

    return dl_val, dl_test


def _init_model(args):

    encoder = ResNet3d18Backbone(**cfg.enc_cfgs)
    decoder = ImPulSeDecoder(**cfg.dec_cfgs)
    corrector = ImPulSeDecoder(**cfg.cor_cfgs)
    impulse = ImPulSe(encoder, decoder, corrector)
    template_generator = TemplateGenerator(**cfg.gen_cfgs)
    model = TIm(impulse, template_generator)

    devices = [int(x) for x in args.gpu.split(",")]
    if len(devices) > 1:
        model = nn.DataParallel(model.cuda(), devices)
    else:
        model = model.cuda()

    return model


def _calculate_bboxes(image_shape, crop_size):
    steps = [np.arange(0, image_shape[i], crop_size[i]).tolist()
        + [image_shape[i]] for i in range(3)]
    begs = [steps[i][:-1] for i in range(3)]
    ends = [steps[i][1:] for i in range(3)]
    bboxes = []
    for i in range(len(begs[0])):
        for j in range(len(begs[1])):
            for k in range(len(begs[2])):
                bboxes.append(np.array([
                    [begs[0][i], ends[0][i]],
                    [begs[1][j], ends[1][j]],
                    [begs[2][k], ends[2][k]],
                ]))
    bboxes = np.stack(bboxes)

    return bboxes


def _sliding_window_predict(model, inputs, grids, window_size):
    resolution = grids.size()[-2:0:-1]
    bboxes = _calculate_bboxes(resolution, window_size)
    output = np.zeros(resolution, dtype=np.uint8)

    for i in range(bboxes.shape[0]):
        grid_patch = grids[
            :,
            bboxes[i, 2, 0]:bboxes[i, 2, 1],
            bboxes[i, 1, 0]:bboxes[i, 1, 1],
            bboxes[i, 0, 0]:bboxes[i, 0, 1],
            :].cuda()

        output_patch, _ = model(inputs, grid_patch)
        output_patch = output_patch.cpu().numpy().squeeze(axis=0)
        output_patch = output_patch.argmax(axis=0).astype(np.uint8)
        output[
            bboxes[i, 0, 0]:bboxes[i, 0, 1],
            bboxes[i, 1, 0]:bboxes[i, 1, 1],
            bboxes[i, 2, 0]:bboxes[i, 2, 1]
        ] = output_patch

    return output


@logger
@torch.no_grad()
def _predict(model, dataloader, output_dir, subset):
    model.eval()
    progress = tqdm(total=len(dataloader))
    os.makedirs(os.path.join(output_dir, subset), exist_ok=True)

    for i, sample in enumerate(dataloader):
        inputs, grids, pids, bboxes, shapes = sample
        pid = pids[0]
        bbox = bboxes[0]
        grids = grids[0][None]
        original_shape = shapes[0]
        inputs = inputs.cuda()
        grids = grids.cuda()

        y_pred_lung = _sliding_window_predict(model, inputs, grids,
            cfg.window_size)
        y_pred = np.zeros(original_shape, dtype=np.uint8)
        y_pred[
            bbox[0, 0]:bbox[0, 1] + 1,
            bbox[1, 0]:bbox[1, 1] + 1,
            bbox[2, 0]:bbox[2, 1] + 1
        ] = y_pred_lung

        y_pred_img = sitk.GetImageFromArray(y_pred.astype(np.uint8))

        sitk.WriteImage(y_pred_img, os.path.join(output_dir, subset,
            f"{pid}_pred.nii.gz"))

        progress.update()

    progress.close()


def _load_weights(weight_path):
    model_weights = torch.load(weight_path, map_location='cuda:0')
    new_model_weights = {}
    for k in model_weights.keys():
        if k.startswith("module."):
            new_k = k[7:]
            new_model_weights[new_k] = model_weights[k]

    return new_model_weights


def main():
    _set_rng_seed(42)

    args = _parse_cmd_args()
    torch.cuda.set_device(int(args.gpu.split(",")[0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cfg
    cfg = import_module(f"configs.{args.cfg}_config")

    dl_val, dl_test = _init_dataloaders()

    model = _init_model(args)
    weight_path = cfg.model_weights_path
    # Using multiple-GPUs to conduct predict
    model_weights = _load_weights(weight_path)
    # model_weights = torch.load(weight_path)
    model.load_state_dict(model_weights)

    output_dir = f"/media/dntech/_mnt_storage/yufei/data/lung_segment/valtest_data/outputs/pred/{args.cfg.upper()}"
    _predict(model, dl_val, output_dir, "val")
    _predict(model, dl_test, output_dir, "test")


if __name__ == "__main__":
    main()
