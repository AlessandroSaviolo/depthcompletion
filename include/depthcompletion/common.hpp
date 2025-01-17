#ifndef JETSON_POSE_COMMON_HPP
#define JETSON_POSE_COMMON_HPP

#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cuda_runtime.h>

// Error checking function for CUDA calls
inline void checkCudaError(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}

// Error checking function for boolean results
inline void checkBoolError(bool result, const char* file, int line, const char* call) {
    if (!result) {
        std::cerr << "Error: " << call << " failed at " << file << ":" << line << std::endl;
        exit(1);
    }
}

// Wrapper macros to use the error checking functions
#define CHECK_CUDA(call) checkCudaError((call), __FILE__, __LINE__)
#define CHECK_BOOL(call) checkBoolError((call), __FILE__, __LINE__, #call)

class Logger : public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
        : reportableSeverity(severity) {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg << std::endl;
    }

    nvinfer1::ILogger::Severity getReportableSeverity() const {
        return reportableSeverity;
    }
};

inline int get_size_by_dims(const nvinfer1::Dims& dims) {
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

inline static float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string& path) {
    return (access(path.c_str(), F_OK) == 0);
}

inline bool IsFile(const std::string& path) {
    if (!IsPathExist(path)) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << path << " not exist\n";
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool IsFolder(const std::string& path) {
    if (!IsPathExist(path)) {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    std::string name;
};

#endif  // JETSON_POSE_COMMON_HPP