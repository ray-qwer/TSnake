import torch
import numpy as np
import torchvision
from torchvision.utils import make_grid, save_image
import sys
sys.path.append('.')
from utils.diff_ras.polygon import SoftPolygon, SoftPolygonPyTorch

def get_rasterizers(inv_smoothness, mode, ras_type='cuda'):
    if isinstance(inv_smoothness, float) or isinstance(inv_smoothness, int):
        inv_smoothness = [inv_smoothness]
    if ras_type == 'cuda':
        raster = SoftPolygon
    elif ras_type == 'pytorch':
        raster = SoftPolygonPyTorch
    return [raster(inv_smoothness=i) for i in inv_smoothness]        


def main():
    rasterizers = get_rasterizers([100, 10, 1, 0.1, 0.01, 0.001], mode='boundary', ras_type='cuda')
    rectangle = torch.tensor([[1.,45.],
                              [45.,1.],
                              [90.,45.],
                              [45.,90.]]).to('cuda')
    rectangle = rectangle.unsqueeze(0)
    mask = torch.cat([r(rectangle-0.5, 100, 100, 1).unsqueeze(1) for r in rasterizers])
    mask = torch.clamp(mask, 0., 1.).detach()
    
    grid = make_grid(mask, nrow=mask.shape[0], padding=10, pad_value=1)
    save_image(grid, "./repo_test/softpolygon.jpg")

    rasterizers = get_rasterizers([100, 10, 1, 0.1, 0.01, 0.001], mode='boundary', ras_type='pytorch')
    mask = torch.cat([r(rectangle-0.5, 100, 100, 1).unsqueeze(1) for r in rasterizers])
    mask = torch.clamp(mask, 0., 1.).detach()
    
    grid = make_grid(mask, nrow=mask.shape[0], padding=10, pad_value=1)
    save_image(grid, "./repo_test/softpolygonpytorch.jpg")



if __name__ == "__main__":
    main()