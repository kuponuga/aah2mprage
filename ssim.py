import os
from SSIM_PIL import compare_ssim
from PIL import Image
import os

image1 = Image.open("/home/hiroshi/Code/pytorch-CycleGAN-and-pix2pix/results/tgse_reg_10000_grey/test_latest/images/tgse014-slice015_real_A.png")
image2 = Image.open("/home/hiroshi/Code/pytorch-CycleGAN-and-pix2pix/results/tgse_reg_10000_grey/test_latest/images/tgse014-slice015_fake_B.png")

# Compare images using OpenCL by default
value = compare_ssim(image1, image2)
print(value)

#  Compare images using CPU-only version
# value = compare_ssim(image1, image2, GPU=False)
# print(value)
