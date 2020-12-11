from __future__ import division
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def predict_mask(in_files, out_files):
    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format('unet_carvana_scale1_epoch5.pth'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device {}".format(device))
    net.to(device=device)
    net.load_state_dict(torch.load('unet_carvana_scale1_epoch5.pth', map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=0.5,
                           out_threshold=0.5,
                           device=device)

        out_fn = out_files[i]
        result = mask_to_image(mask)
        result.save(out_files[i])

        logging.info("Mask saved to {}".format(out_files[i]))

