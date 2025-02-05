# Depth Completion C++ Library

A lightweight and efficient library for enhancing noisy or incomplete depth data from sensors like RGB-D cameras, enabling reliable perception for robotics applications.

## Motivation

During the development of a reactive collision avoidance framework for agile navigation, we encountered significant challenges with raw depth data from RGB-D sensors. This data was often noisy, incomplete, and lacked sufficient range, particularly in cluttered and dynamic environments. To overcome these limitations, we integrated monocular depth estimation with a depth completion pipeline, enabling the creation of accurate, dense, and extended depth maps that significantly improved perception capabilities.

## How it works

Our depth completion approach begins with input from an RGB-D sensor, which provides two data streams: an RGB image capturing the visual scene and a sparse depth map containing distance measurements. However, due to sensor limitations, the depth map often has gaps and noise, particularly near reflective surfaces or object edges.

To address these issues, we use a monocular depth estimation network to generate a relative depth map from the RGB image. This network produces a dense depth map that fills in the missing regions. However, these predicted depths are relative and not in absolute metric units. To align the predicted depth map with the sparse absolute depth from the sensor, we apply a second-order polynomial model. This model minimizes the error between the predicted depth and sensor data in regions where both are valid, ensuring the predictions are correctly scaled.

The aligned depth map is then refined using the polynomial model, resulting in a completed absolute depth map. This map combines the sensor's accurate absolute data with the network's ability to generalize across the scene, filling in gaps and extending the depth range. The result is a dense, high-quality depth map that significantly enhances perception for robotics tasks such as obstacle detection, planning, and navigation.

## Features
- Combines raw RGB-D depth maps with monocular depth estimation outputs for enhanced depth accuracy.
- Optimized for fast inference, enabling use on lightweight robotics platforms.
- Easily adaptable to different environments and sensor setups.
- A basic ROS2 nodelet is included as an example and for testing purposes.

---

## Installation

To install dependencies (e.g., PyTorch, TensorRT), follow the instructions in [`install_dependencies.md`](https://github.com/AlessandroSaviolo/depthcompletion/blob/main/install_dependencies.md).

---

## ROS2 Testing
> **Note:** ROS2 is optional. If ROS2 integration is not required, you can remove ROS2 dependencies and use the library independently.

### 1. Create a Workspace
```
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
```

### 2. Clone and Build the Package
Clone the repositories:
```
git clone https://github.com/AlessandroSaviolo/realsense-cpp
```
```
git clone https://github.com/AlessandroSaviolo/depthcompletion
```
Build the workspace using `colcon`:
```
cd ~/ros_ws
colcon build --symlink-install
```
Source the workspace:
```
source install/local_setup.bash
```

### 3. Export PyTorch model to TensorRT format
Run on your laptop:
```
cd ~/depthcompletion/scripts
python export_onnx.py
```
Run on your embedded device:
```
python export_trt.py
```

### 4. Launch and Visualize
Launch the ROS2 nodelet:
```
ros2 launch depthcompletion depthcompletion.launch.py
```
Visualize the output using `rqt_image_view`:
```
ros2 run rqt_image_view rqt_image_view
```

---

## Tested Environment
This library has been tested in the following setup:
- **Hardware:** NVIDIA Orin 16GB, Intel RealSense 455
- **Librealsense Version:** v2.55.1
- **ROS2 Distribution:** Humble

---

## Contributing
Contributions are welcome! You can:
- Extend functionality.
- Improve documentation.
- Fix bugs or issues.
Feel free to submit pull requests or open issues.

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
