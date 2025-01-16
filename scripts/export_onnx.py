import sys
import os
import numpy as np
import torch
import pytorch_lightning
from onnx import checker, load

from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

#----------------------------------------------------------------------------

class MonocularDepthEstimationModel(pytorch_lightning.LightningModule):
    def __init__(self, encoder_type, model_path, device):
        super().__init__()
        self.model = DepthAnythingV2(encoder=f"{encoder_type}", features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(model_path, map_location=device))

    def forward(self, inp):
        _, out = self.model(inp)
        return out

#----------------------------------------------------------------------------

def export_model_to_onnx(da_encoder_type, model_torch_path, model_onnx_path, input_shape):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Found device', device)

    pytorch_lightning.seed_everything(17)
    sample_input = torch.rand(1, 3, input_shape[0], input_shape[1]).float().to(device)
    
    model = MonocularDepthEstimationModel(da_encoder_type, model_torch_path, device)
    model = model.eval().to(device)

    try:
        model.to_onnx(
            model_onnx_path,
            sample_input,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input':  {0: 'batch_size'}, 
                'output': {0: 'batch_size'}
            }
        )
        checker.check_model(load(model_onnx_path))
        print('ONNX model export and validation successful.')
    except Exception as e:
        print(f'Error during model export or validation: {e}')

#----------------------------------------------------------------------------

def get_multiple_of_14(prompt):
    while True:
      try:
          value = int(input(prompt))
          if value % 14 == 0:
              return value
          else:
              print("Value must be a multiple of 14. Please try again.")
      except ValueError:
          print("Invalid input. Please enter an integer.")

#----------------------------------------------------------------------------

def main():
    print('Exporting Depth Anything v2 to ONNX format.')
    print('Make sure to run this ON YOUR LAPTOP.')

    # input_shape = (238, 322)
    width = get_multiple_of_14("Enter image width (multiple of 14): ")
    height = get_multiple_of_14("Enter image height (multiple of 14): ")
    print(f"Image dimensions: {width}x{height}")
    input_shape = (height, width)

    da_encoder_type = 'vits'
    da_path = "/".join(sys.path[0].split("/")[:-1]) + "/scripts/"
    model_name = f"depth_anything_v2/checkpoints/depth_anything_v2_{da_encoder_type}"
    model_torch_path = da_path + model_name + ".pth"
    model_onnx_path = da_path + model_name + ".onnx"

    export_model_to_onnx(da_encoder_type, model_torch_path, model_onnx_path, input_shape)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
