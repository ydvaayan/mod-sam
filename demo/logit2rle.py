import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from segment_anything.utils.amg import mask_to_rle_pytorch
import torch

parser = argparse.ArgumentParser(
    description="Convert demo output to RLE format"
)
parser.add_argument(
    "--input_path", type=str, required=True, help="The path to the input json."
)
parser.add_argument(
    "--output_path", type=str, required=True, help="The path to the output json."
)
parser.add_argument(
    "--image_id", type=str, required=True, help="id of image whose mask is being changed"
)
parser.add_argument(
    "--ann_id", type=str, required=True, help="id of mask that is being changed."
)

if __name__ == "__main__":
    args = parser.parse_args()

    x=json.load(open(args.input_path))

    h=x["height"]
    w=x["width"]

    y=[]
    for i in range(len(x["segmentation"])):
        if x["segmentation"][str(i)]>0:
            y.append(1)
        else:
            y.append(0)
    y=np.array(y)
    y=np.reshape(y,(h,w))
    rle_masks=[]
    y_tensor=torch.tensor(y,dtype=torch.int32,device='cpu')
    y_tensor=y_tensor.unsqueeze(0)
    rle_masks=mask_to_rle_pytorch(y_tensor)
    data={"annotations":[]}
    new_ann={"segmentation":rle_masks,"image_id":args.image_id,"annotation_id":args.ann_id}
    data["annotations"].append(new_ann)
    
    with open(args.output_path, 'w') as f:
        json.dump(data, f)
    # plt.imshow(y, cmap='gray')
    # plt.title('Binary Mask')
    # plt.colorbar()
    # plt.show()


    
