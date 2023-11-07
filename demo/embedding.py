import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
import argparse

parser = argparse.ArgumentParser(
    description="Convert demo output to RLE format"
)
parser.add_argument(
    "--input_path", type=str, required=True, help="The path to the input image."
)
parser.add_argument(
    "--output_path", type=str, required=True, help="The path to the output embedding."
)
parser.add_argument(
    "--checkpoint", type=str, default="../sam_vit_h_4b8939.pth", help="id of image whose mask is being changed"
)
parser.add_argument(
    "--model_type", type=str, default= "vit_h", help="id of mask that is being changed."
)

if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint = args.checkpoint
    model_type = args.model_type
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device='cpu')
    predictor = SamPredictor(sam)

    image = cv2.imread(args.input_path)
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    np.save(args.output_path, image_embedding)