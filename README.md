# Depth Completion C++ Library

A lightweight and efficient library for enhancing noisy or incomplete depth data from sensors like RGB-D cameras, enabling reliable perception for robotics applications.

## Motivation

In robotics, accurate depth information is crucial for tasks like obstacle avoidance, path planning, and reactive control. However, sensors such as RGB-D cameras often produce noisy, incomplete, or limited-range depth maps due to hardware and environmental limitations. Depth completion bridges this gap by combining sensor data with advanced algorithms to create dense and accurate depth maps.

This library is designed to be simple, efficient, and adaptable to real-world scenarios. It focuses solely on depth completion, providing easy integration into robotics pipelines. Contributions to improve or extend functionality are welcome!

## Features
- Combines raw RGB-D depth maps with outputs from monocular depth estimation networks.
- Optimized for fast inference, enabling use on lightweight robotics platforms.
- Easily adaptable to different environments and sensor setups.
- A basic ROS2 nodelet is included as an example and for testing purposes.

---

## Installation

### 1. Installing Pytorch on Jetpack 5
```
sudo apt-get -y install python3-pip libopenblas-dev
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```
```
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.26.1
python3 -m pip install typing-extensions --upgrade
python3 -m pip install --no-cache $TORCH_INSTALL
```

Verify installation:
```
python
```
```
import torch
torch.cuda.is_available()
```
**NOTE:** if cuda is not available install it using the following command on Jetson
```
sudo apt update
sudo apt-get install nvidia-jetpack

# add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH'>>~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH'>>~/.bashrc
source ~/.bashrc
```

### 2. Installing Torchvision from source (make sure to match your torch version with the torchvision branch)
```
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```
```
git clone https://github.com/pytorch/vision torchvision
```
```
cd torchvision
git checkout release/0.15
```
```
python3 setup.py install --user
```

### 3. Install PyCuda from source
```
cd ..
git clone https://github.com/inducer/pycuda.git
```
```
cd pycuda
git submodule update --init
python3 configure.py
```
```
sudo make install
```

### 4. Install Tensorrt
```
sudo apt update
```
```
sudo apt install nvidia-tensorrt

# Add tensorrt to PATH
echo 'export PATH=$PATH:/usr/src/tensorrt/bin' >> ~/.bashrc
source ~/.bashrc
```

Check Tensorrt installation:
```
trtexec --help
```

---

## Export ONNX to TensorRT
```
cd ~/depthcompletion/scripts
python export_trt.py
```

---

## ROS2 Testing
> **Note:** ROS2 is not required to use this library. If you do not need ROS2 integration, you can remove ROS2 dependencies.

### 1. Create a Workspace
```bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
```

### 2. Clone and Build the Package
Build the workspace using `colcon`:
```
git clone https://github.com/AlessandroSaviolo/realsense-cpp
```
```
git clone https://github.com/AlessandroSaviolo/depthcompletion
```
Source the workspace:
```
cd ~/ros_ws
colcon build --symlink-install
source install/local_setup.bash
```

### 3. Launch and Visualize
Launch the ROS2 nodelet:
```bash
ros2 launch depthcompletion depthcompletion.launch.py
```
Visualize the output using `rqt_image_view`:
```bash
ros2 run rqt_image_view rqt_image_view
```

---

## Tested Environment
This code was tested on the following setup:
- **Hardware:** NVIDIA Orin 16GB, Intel RealSense 455
- **Librealsense Version:** v2.55.1
- **ROS2 Distribution:** Humble

---

## Contributing
Contributions are encouraged! Feel free to submit pull requests to:
- Extend functionality.
- Improve documentation.
- Fix bugs or issues.

---

## License
This project is licensed under MIT license.

```license
Copyright (c) 2024 AlessandroSaviolo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## Citation

This repository includes code that supports the paper **"Reactive Collision Avoidance for Safe Agile Navigation"**. 
If you use this library in your research or academic projects, please cite the following:
```bibtex
@article{saviolo2024reactive,
  title={Reactive Collision Avoidance for Safe Agile Navigation},
  author={Saviolo, Alessandro and Picello, Niko and Verma, Rishabh and Loianno, Giuseppe},
  journal={arXiv preprint arXiv:2409.11962},
  year={2024}
}
```
