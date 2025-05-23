#pragma once

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>

#include "common.hpp"
#include "half.hpp"

// Static configuration for data precision
static const nvinfer1::DataType DATATYPE = nvinfer1::DataType::kHALF;
using FloatPrecision = std::conditional_t<DATATYPE == nvinfer1::DataType::kHALF, __half, float>;

// DepthCompletionEngine: Responsible for loading the TensorRT engine and running inference
class DepthCompletionEngine {
public:
    DepthCompletionEngine();
    DepthCompletionEngine(const std::string &engine_file_path, int batch_size, 
                          int input_height, int input_width, int input_channels);
    ~DepthCompletionEngine();

    void init(const std::string &engine_file_path, int batch_size, 
              int input_height, int input_width, int input_channels);
    cv::Mat predict(cv::Mat &frame);

private:
    // TensorRT objects
    nvinfer1::ICudaEngine *_engine = nullptr;
    nvinfer1::IRuntime *_runtime = nullptr;
    nvinfer1::IExecutionContext *_context = nullptr;

    // CUDA stream
    cudaStream_t _stream = nullptr;

    // Logger for TensorRT
    Logger _gLogger{nvinfer1::ILogger::Severity::kERROR};

    // Binding information
    std::vector<Binding> _input_bindings;
    std::vector<Binding> _output_bindings;
    std::vector<void *> _gpu_ptrs;

    int _num_bindings = 0;
    int _num_inputs = 0;
    int _num_outputs = 0;

    // Input/output sizes and dimensions
    int _batch_size = 0;
    int _input_height = 0, _input_width = 0, _input_channels = 0;
    int _output_height = 0, _output_width = 0, _output_channels = 0;
    size_t _input_size = 0, _output_size = 0;

    // Normalization parameters
    cv::Scalar _imagenet_mean{0.485, 0.456, 0.406};
    cv::Scalar _imagenet_std{0.229, 0.224, 0.225};
    cv::Mat _imagenet_mean_mat, _imagenet_std_mat;
};

// Constructor: Initializes normalization parameters
DepthCompletionEngine::DepthCompletionEngine() = default;

DepthCompletionEngine::DepthCompletionEngine(const std::string &engine_file_path, int batch_size, 
                                             int input_height, int input_width, int input_channels) {
    init(engine_file_path, batch_size, input_height, input_width, input_channels);
}

// Destructor: Releases TensorRT and GPU resources
DepthCompletionEngine::~DepthCompletionEngine() {
    if (_context) _context->destroy();
    if (_engine) _engine->destroy();
    if (_runtime) _runtime->destroy();
    if (_stream) cudaStreamDestroy(_stream);

    for (auto &ptr : _gpu_ptrs) {
        CHECK_CUDA(cudaFree(ptr));
    }
}

// Initialize the TensorRT engine
void DepthCompletionEngine::init(const std::string& engine_file_path, const int batch_size, 
                     const int input_height, const int input_width, const int input_channels) {

    // Load engine file into memory
    std::cout << "[DepthCompletionEngine] Loading engine from file: " << engine_file_path << std::endl;
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);

    file.read(trtModelStream, size);
    file.close();

    std::cout << "[DepthCompletionEngine] Initializing TensorRT plugins..." << std::endl;
    initLibNvInferPlugins(&_gLogger, "");

    std::cout << "[DepthCompletionEngine] Creating TensorRT runtime..." << std::endl;
    _runtime = nvinfer1::createInferRuntime(_gLogger);
    assert(this->runtime != nullptr);

    std::cout << "[DepthCompletionEngine] Deserializing CUDA engine..." << std::endl;
    _engine = _runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(this->engine != nullptr);

    delete[] trtModelStream;

    std::cout << "[DepthCompletionEngine] Creating execution context..." << std::endl;
    _context = _engine->createExecutionContext();
    assert(this->context != nullptr);

    std::cout << "[DepthCompletionEngine] Creating CUDA stream..." << std::endl;
    CHECK_CUDA(cudaStreamCreate(&_stream));

    // Process bindings
    _num_bindings = _engine->getNbBindings();
    std::cout << "[DepthCompletionEngine] Number of bindings: " << _num_bindings << std::endl;

    for (int i = 0; i < _num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype = _engine->getBindingDataType(i);
        std::string name = _engine->getBindingName(i);
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        if (_engine->bindingIsInput(i)) {
            std::cout << "[DepthCompletionEngine] Binding " << i << " is an input: " << name << std::endl;
            dims = _engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            _input_bindings.push_back(binding);
            _num_inputs += 1;
            CHECK_BOOL(_context->setBindingDimensions(i, dims));
        } else {
            std::cout << "[DepthCompletionEngine] Binding " << i << " is an output: " << name << std::endl;
            dims = _context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            _output_bindings.push_back(binding);
            _num_outputs += 1;
        }
        std::cout << "[DepthCompletionEngine] Binding name: " << name << ", size: " << binding.size << ", dsize: " << binding.dsize << std::endl;
    }
    std::cout << "[DepthCompletionEngine] Initialization complete." << std::endl;

    std::cout << "[DepthCompletionEngine] Allocating gpu memory..." << std::endl << std::flush;
    for (auto& bindings : _input_bindings) {
        void* gpu_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK_CUDA(cudaMalloc(&gpu_ptr, size));
        _gpu_ptrs.push_back(gpu_ptr);
        std::cout << "[DepthCompletionEngine] Allocated input binding gpu memory: " << gpu_ptr << " of size: " << size << std::endl << std::flush;
    }
    for (auto& bindings : _output_bindings) {
        void *gpu_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK_CUDA(cudaMalloc(&gpu_ptr, size));
        _gpu_ptrs.push_back(gpu_ptr);
        std::cout << "[DepthCompletionEngine] Allocated output binding gpu memory: " << gpu_ptr << " of size: " << size << std::endl << std::flush;
    }

    // Set sizes
    _batch_size      =  _input_bindings[0].dims.d[0];
    _input_channels  =  _input_bindings[0].dims.d[1];
    _input_height    =  _input_bindings[0].dims.d[2];
    _input_width     =  _input_bindings[0].dims.d[3];
    _output_channels = _output_bindings[0].dims.d[1];
    _output_height   = _output_bindings[0].dims.d[2];
    _output_width    = _output_bindings[0].dims.d[3];
    _input_size      =  _input_bindings[0].dims.d[0] *  _input_bindings[0].dims.d[1] *  _input_bindings[0].dims.d[2] *  _input_bindings[0].dims.d[3];
    _output_size     = _output_bindings[0].dims.d[0] * _output_bindings[0].dims.d[1] * _output_bindings[0].dims.d[2] * _output_bindings[0].dims.d[3];

    // Set normalization parameters
    cv::Size input_size(_input_bindings[0].dims.d[3], _input_bindings[0].dims.d[2]);
    cv::Size output_size(_output_bindings[0].dims.d[3], _output_bindings[0].dims.d[2]);

    _imagenet_mean_mat = cv::Mat(input_size, CV_32FC3, _imagenet_mean);
    _imagenet_std_mat  = cv::Mat(input_size, CV_32FC3, _imagenet_std);
}

// Run inference on a given frame
cv::Mat DepthCompletionEngine::predict(cv::Mat &frame) {

    // Allocate CPU memory for input and output
    __half* input   = new __half[_input_size]();
    __half* output  = new __half[_output_size]();
    float* output32 = new float[_output_size]();

    // Rearrange image data from [W, H, C] to [C, H, W] and flatten it
    unsigned int input_index = 0;
    for (int c = 0; c < _input_channels; c++) {
        for (int i = 0; i < _input_height; i++) {
            for (int j = 0; j < _input_width; j++) {
                input[input_index++] = fp16::__float2half(
                    ((frame.at<cv::Vec3b>(i, j)[c] / 255.f) - _imagenet_mean[c]) / _imagenet_std[c]);
            }
        }
    }

    // DMA the input to the GPU, execute the batch asynchronously, and DMA it back
    CHECK_CUDA(cudaMemcpyAsync(_gpu_ptrs[0], input, _input_size * sizeof(__half), cudaMemcpyHostToDevice, _stream));
    CHECK_BOOL(_context->enqueueV2(_gpu_ptrs.data(), _stream, nullptr));
    CHECK_CUDA(cudaMemcpyAsync(output, _gpu_ptrs[1], _output_size * sizeof(__half), cudaMemcpyDeviceToHost, _stream));
    CHECK_CUDA(cudaStreamSynchronize(_stream));

    // Postprocess
    for (int i = 0; i < _output_size; i++)
        output32[i] = fp16::__half2float(output[i]);
    cv::Mat result(_output_height, _output_width, CV_32F, output32);

    // Clean up
    delete[] input;
    delete[] output;

    return result.clone();  // Clone to ensure valid memory after cleanup
}