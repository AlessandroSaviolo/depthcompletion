import sys
import os
import cv2
from PIL import Image
import numpy as np
import torch
import tensorrt as trt 
import pycuda.driver
import pycuda.autoinit

#----------------------------------------------------------------------------

def test_trt(trt_path, sample_path, input_shape, output_shape, batch_size=1):

    f = open(trt_path, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    sample_input = np.random.rand(batch_size, 3, input_shape[0], input_shape[1]).astype(np.float16)
    sample_output = np.empty([batch_size, output_shape[0]*output_shape[1]], dtype=np.float16)

    d_input = pycuda.driver.mem_alloc(1 * sample_input.nbytes)
    d_output = pycuda.driver.mem_alloc(1 * sample_output.nbytes)
    bindings = [int(d_input), int(d_output)]
    stream = pycuda.driver.Stream()

    def predict(batch):
        batch = np.ascontiguousarray(batch)
        pycuda.driver.memcpy_htod_async(d_input, batch, stream)
        context.execute_async_v2(bindings, stream.handle, None)
        pycuda.driver.memcpy_dtoh_async(sample_output, d_output, stream)
        stream.synchronize()
        return sample_output

    imagenet_mean = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis]
    imagenet_std = np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]

    input = cv2.imread(sample_path)
    input = cv2.resize(input, (input_shape[1], input_shape[0]))
    input = input.transpose(2, 0, 1) 
    input = input / 255.0
    input = (input - imagenet_mean) / imagenet_std
    input = np.expand_dims(input, 0)
    input = input.astype(np.float16)

    pred = predict(input)

    # pred = 1. / (pred + 1e-3)
    pred = pred.reshape(output_shape)
    pred = ((pred - np.min(pred)) / (np.max(pred) - np.min(pred)) * 255).astype(np.uint8)
    pred_image = Image.fromarray(pred, 'L')
    pred_image.save(sample_path.replace('.jpg', '_pred.jpg'))

    frequency_hist = []
    for i in range(11):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        predict(sample_input)
        ender.record()
        torch.cuda.synchronize()
        if i:
            frequency_hist.append(1. / (starter.elapsed_time(ender) / 1000))
    print("Inference [Hz]:", np.mean(frequency_hist))

#----------------------------------------------------------------------------

def main():
    print('Testing Depth Anything v2 with TRT format.')

    da_path = "/".join(sys.path[0].split("/")[:-1]) + "/scripts/"
    model_name = f"depth_anything_v2/checkpoints/depth_anything_v2_vits"
    model_trt_path = da_path + model_name + ".trt"
    sample_path = da_path + 'depth_anything_v2/assets/examples/demo01.jpg'

    input_shape  = (238, 322)
    output_shape = (238, 322)

    test_trt(model_trt_path, sample_path, input_shape, output_shape)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
