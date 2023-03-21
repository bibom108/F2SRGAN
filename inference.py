from F2SRGAN.model import *
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import imageio
import math
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPSCALE_FACTOR = 4

def tensors_to_imgs(x):
    for i in range(len(x)):
        x[i] = x[i].squeeze(0).data.cpu().numpy()
        x[i] = x[i].clip(0, 255)#.round()
        x[i] = x[i].transpose(1, 2, 0).astype(np.uint8)
    return x


def imgs_to_tensors(x):
    for i in range(len(x)):
        x[i] = x[i].transpose(2, 0, 1)
        x[i] = np.expand_dims(x[i], axis=0)
        x[i] = torch.Tensor(x[i].astype(float))
    return x


def calc_psnr(sr, hr, scale, rgb_range=255):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def swiftsrgan_psnr_testing(_path, model_path, ds):
    netG = Generator().to(DEVICE)
    netG.load_state_dict(torch.load(model_path, map_location=DEVICE)['model'])
    netG.eval()

    with torch.no_grad():
        valing_results = {
            "mse": 0,
            "psnr": 0,
            "batch_sizes": 0,
            "ssim": 0,
            "ssims": 0,
        }
        nums = 0
        for name in os.listdir(_path):
            full_path = _path + "/" + name
            hr_image = Image.open(full_path).convert('RGB')
            image_width = (hr_image.width // 4) * 4
            image_height = (hr_image.height // 4) * 4
            hr_scale = transforms.Resize((image_height, image_width), interpolation=Image.BICUBIC)
            lr_scale = transforms.Resize((image_height // 4, image_width // 4), interpolation=Image.BICUBIC)
            lr_image = lr_scale(hr_image)
            hr_image = hr_scale(hr_image)
            lr_image = np.asarray(lr_image)
            hr_image = np.asarray(hr_image)
            
            [lr_image] = imgs_to_tensors([lr_image])
            [hr_image] = imgs_to_tensors([hr_image])

            out = netG(lr_image)
            valing_results["psnr"] += calc_psnr(out, hr_image, 1)
            [out] = tensors_to_imgs([out])
            [hr_image] = tensors_to_imgs([hr_image])

            imageio.imwrite('./PIRM2018/your_results' + "/" + name, out)
            imageio.imwrite("./PIRM2018/self_validation_HR" + "/" + name, hr_image)
            nums += 1
    
    return valing_results["psnr"]/nums

dataset = ["BSDS100"]
model_path = f"./pretrain_weight/F2SRGAN_{UPSCALE_FACTOR}x.pt"

for ds in dataset:
    psnr = swiftsrgan_psnr_testing("./SR_testing_datasets/" + ds, model_path, ds)
    print(f'PSNR in {ds} = {psnr}')

