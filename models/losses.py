import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, output, target):
        eps = 1e-8
        fg_output = torch.softmax(output, dim=1)[:, 1:, ...]
        fg_target = F.one_hot(target, self.num_classes)[..., 1:]
        fg_target = fg_target.permute(0, 4, 1, 2, 3)
        dice_loss = 1 - 2 * (fg_output * fg_target).sum() \
                    / (fg_output.sum() + fg_target.sum() + eps)

        return dice_loss


class SegLoss(nn.Module):

    def __init__(self, w_ce, w_dice, num_classes):
        super().__init__()

        self.w_ce = w_ce
        self.w_dice = w_dice
        self.ce_fn = nn.CrossEntropyLoss()
        self.dice_fn = DiceLoss(num_classes)

    def forward(self, output, target):
        ce_loss = self.ce_fn(output, target)
        dice_loss = self.dice_fn(output, target)
        total_loss = self.w_ce * ce_loss + self.w_dice * dice_loss

        return total_loss


class DefLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, deformation):
        def_loss = torch.norm(deformation)

        return def_loss


class TemDefLoss(nn.Module):

    def __init__(self, w_ce, w_dice, w_def, num_classes):
        super().__init__()

        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_def = w_def
        self.ce_fn = nn.CrossEntropyLoss()
        self.dice_fn = DiceLoss(num_classes)
        self.def_fn = DefLoss()

    def forward(self, output, target, deformation):
        ce_loss = self.ce_fn(output, target)
        dice_loss = self.dice_fn(output, target)
        def_loss = self.def_fn(deformation)
        total_loss = self.w_ce * ce_loss + self.w_dice * dice_loss + self.w_def * def_loss

        return total_loss



