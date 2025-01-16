import sys
import os
import tensorrt as trt 
import pycuda.driver
import pycuda.autoinit

#----------------------------------------------------------------------------

def export_model_to_trt(onnx_path, trt_path):
    if os.path.isfile(trt_path):
        while(1):
            ans = input('Found existing TRT model. Remove? (Y or N)')
            if ans == 'Y' or 'YES' or 'yes' or 'y':
                break
            elif ans == 'N' or 'NO' or 'no' or 'n':
                return
            else:
                continue

    print('Converting model to TRT.')
    cmd = '/usr/src/tensorrt/bin/trtexec' +\
        ' --onnx=' + onnx_path + \
        ' --saveEngine=' + trt_path + \
        ' --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16'
    os.system(cmd)

#----------------------------------------------------------------------------

def main():
    print('Exporting Depth Anything v2 to TRT format.')

    da_path = "/".join(sys.path[0].split("/")[:-1]) + "/scripts/"
    model_name = f"depth_anything_v2/checkpoints/depth_anything_v2_vits"
    model_onnx_path = da_path + model_name + ".onnx"
    model_trt_path = da_path + model_name + ".trt"

    export_model_to_trt(model_onnx_path, model_trt_path)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()