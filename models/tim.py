from tempfile import template
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from .conv import ConvBlock3d, ConvLayer3d
from random import seed
import gc

INTERPOLATIONS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,

}


class _LocalNorm3d(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        output = self.norm(x)
        output = output.permute(0, 4, 1, 2, 3)

        return output.contiguous()


class ImPulSeDecoder(nn.Module):

    def __init__(self, num_channels, num_classes, num_layers, drop_prob):
        super().__init__()
        # MLP made up of consecutive 1x1x1 conv layers
        self.conv_blocks = nn.ModuleList([ConvBlock3d(num_channels[i],
            num_channels[i + 1], 1, _LocalNorm3d, num_layers, drop_prob)
            for i in range(len(num_channels) - 1)])
        self.output_layer = ConvLayer3d(num_channels[-1], num_classes, 1,
            True, None, None, drop_prob)

    def forward(self, x):
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x)

        output = self.output_layer(x)

        return output


class ImPulSe(nn.Module):

    def __init__(self, encoder, decoder, corrector):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.corrector = corrector

    def make_point_encoding(self, features, grids):
        # interpolate features at continuous locations and concatenate

        point_encodings = [F.grid_sample(features[i], grids, "bilinear",
            align_corners=True) for i in range(len(features))]

        point_encodings.insert(0, grids.permute(0, 4, 1, 2, 3))
        point_encodings = torch.cat(point_encodings, dim=1)
        point_encodings = point_encodings.permute(0, 1, 4, 3, 2)

        return point_encodings

    def forward(self, x, grids):
        # encode spatial features
        features = self.encoder(x)

        # calculate point encodings with features and coordinates
        point_encodings = self.make_point_encoding(features, grids)
        
        # decode features and calculate SDF
        deformation_field = self.decoder(point_encodings)
        correction_field = self.corrector(point_encodings)

        return deformation_field, correction_field

class TemplateGenerator(nn.Module):

    def __init__(self, num_channels, num_layers, num_classes, latent_dim):
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.vec = torch.nn.Parameter(torch.rand(1, latent_dim, 1, 1, 1), requires_grad = True)


        for i in range(len(num_channels) - 1):
            in_channels = num_channels[i]
            out_channels = num_channels[i + 1]
            self.conv_blocks.append(
                ConvBlock3d(in_channels, out_channels, 3,
                            nn.BatchNorm3d, num_layers, 0, True, False)
            )

        self.output_layer = ConvLayer3d(num_channels[-1], num_classes, 1,
                                        True, nn.BatchNorm3d, None)

        
    def forward(self, batch_size):

        latent_code = []
        for i in range(batch_size):
            latent_code.append(self.vec)
        
        latent_code = torch.cat(latent_code, dim=0)
 
        x = F.interpolate(latent_code, scale_factor=2, mode="trilinear",
                                     align_corners=True)
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x)

        output = self.output_layer(x)
    
        return output


class TIm(nn.Module):

    def __init__(self, impulse, template_generator, interpolation="bspline"):
        super().__init__()

        self.impulse = impulse
        self.template_generator = template_generator
        self.interpolation = interpolation
        self.Tanh = nn.Tanh()

        self.impulse.decoder.output_layer[0].weight.data.zero_()
        self.impulse.decoder.output_layer[0].bias.data.zero_()
        self.template_generator.vec.requires_grad = True
        

    def forward(self, x, grids):

        deformation, correction = self.impulse(x, grids)
        deformation = deformation.permute(0, 4, 3, 2, 1)
        
        deformation_field = grids + deformation 
        deformation_field = self.Tanh(deformation_field)
        template = self.template_generator(grids.shape[0])

        output = F.grid_sample(template, deformation_field, mode='bilinear', align_corners=True)
        output = output.permute(0, 1, 4, 3, 2)

        output = output + correction

        return output, deformation