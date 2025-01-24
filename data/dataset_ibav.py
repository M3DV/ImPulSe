import os
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _collate_train(samples):
    inputs = torch.stack([x["inputs"] for x in samples])
    targets = torch.cat([x["targets"] for x in samples]).squeeze_(dim=1)
    grids = torch.stack([x["grids"] for x in samples])

    return inputs, targets, grids


def _collate_infer(samples):
    inputs = torch.stack([x["inputs"] for x in samples])
    grids = [x["grids"] for x in samples]
    pids = [x["pid"] for x in samples]
    bboxes = np.stack([x["lung_bbox"] for x in samples])
    shapes = [x["raw_res"] for x in samples]

    return inputs, grids, pids, bboxes, shapes



class LungSegmentDataset(Dataset):

    def __init__(self, df, data_dir, transforms, subset):
        self.df = df.loc[df.subset == subset].reset_index(drop=True)
        self.data_dir = data_dir
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)

    def _apply_transforms(self, data):
        for t in self.transforms:
            data = t(data)

        return data

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, f"{self.df.pid[idx]}")

        data = {}
        data["pid"] = self.df.pid[idx]
        
        img_image = sitk.ReadImage(os.path.join(data_path, f"{self.df.pid[idx]}_image.nii.gz"))
        data['image'] =  sitk.GetArrayFromImage(img_image)

        airway_image = sitk.ReadImage(os.path.join(data_path, f"{self.df.pid[idx]}_airway.nii.gz"))
        data['airway'] =  sitk.GetArrayFromImage(airway_image)

        artery_image = sitk.ReadImage(os.path.join(data_path, f"{self.df.pid[idx]}_artery.nii.gz"))
        data['artery'] =  sitk.GetArrayFromImage(artery_image)

        vein_image = sitk.ReadImage(os.path.join(data_path, f"{self.df.pid[idx]}_vein.nii.gz"))
        data['vein'] =  sitk.GetArrayFromImage(vein_image)
        
        lungsegment_image = sitk.ReadImage(os.path.join(data_path, f"{self.df.pid[idx]}_lungsegment.nii.gz"))
        data['lungsegment'] =  sitk.GetArrayFromImage(lungsegment_image)
        
        data = self._apply_transforms(data)
        
        return data

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0,
            mode="train"):
        if mode == "train":
            return DataLoader(dataset, batch_size, shuffle,
                num_workers=num_workers,
                collate_fn=_collate_train)
        elif mode == "infer":
            return DataLoader(dataset, batch_size, shuffle,
                num_workers=num_workers,
                collate_fn=_collate_infer)

