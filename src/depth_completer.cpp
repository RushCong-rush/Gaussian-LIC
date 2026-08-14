/*
 * Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <fstream>
#include <stdexcept>
#include "cuda_runtime_api.h"
#include "depth_completer.h"

void DepthCompleter::Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept 
{
    if (severity <= nvinfer1::ILogger::Severity::kWARNING) 
    {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

template <typename T> 
void DepthCompleter::InferDeleter::operator()(T* obj) const 
{ 
    if (obj) delete obj; 
}

DepthCompleter::DepthCompleter(const std::string& enginePath, 
                               int inputWidth, int inputHeight)
    : mInputWidth(inputWidth), 
      mInputHeight(inputHeight) 
{
    initEngine(enginePath);
}

DepthCompleter::~DepthCompleter() 
{
    if (mStream) cudaStreamDestroy(mStream);
    mContext.reset();
    mEngine.reset();
    mRuntime.reset();
    
    for (auto& buf : mDeviceBuffers) 
    {
        if (buf) cudaFree(buf);
    }
}

cv::Mat DepthCompleter::complete(const cv::Mat& rgbImage, const cv::Mat& depthImage) 
{
    cv::Mat processedRgb, processedDepth;
    // rgbImage.convertTo(processedRgb, CV_32F, 1.0f / 255.0f);
    processedRgb = rgbImage;
    depthImage.convertTo(processedDepth, CV_32F, 1.0f / 200.0f);
    prepareInputs(processedRgb, processedDepth);
    
    if (!mContext->enqueueV3(mStream) || cudaStreamSynchronize(mStream) != cudaSuccess)
    {
        throw std::runtime_error("Failed to execute inference");
    }
    
    return processOutput();
}

void DepthCompleter::initEngine(const std::string& enginePath) 
{
    auto engineData = readFile(enginePath);
    initLibNvInferPlugins(&mLogger, "");
    mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!mEngine) throw std::runtime_error("Failed to deserialize engine");
    mContext.reset(mEngine->createExecutionContext());
    if (!mContext) throw std::runtime_error("Failed to create execution context");
    if (cudaStreamCreate(&mStream) != cudaSuccess)
        throw std::runtime_error("Failed to create CUDA stream");

    mContext->setInputShape("rgb", nvinfer1::Dims4(1, 3, mInputHeight, mInputWidth));
    mContext->setInputShape("depth", nvinfer1::Dims4(1, 1, mInputHeight, mInputWidth));
    mContext->setInputShape("mask", nvinfer1::Dims4(1, 1, mInputHeight, mInputWidth));
    
    allocateBuffers();
}

std::vector<char> DepthCompleter::readFile(const std::string& filename) 
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Unable to open file: " + filename);
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) throw std::runtime_error("Failed to read file: " + filename);
    
    return buffer;
}

void DepthCompleter::allocateBuffers() 
{
    const int numTensors = mEngine->getNbIOTensors();
    mDeviceBuffers.resize(numTensors, nullptr);
    mHostBuffers.resize(numTensors);
    mTensorNames.resize(numTensors);
    
    for (int i = 0; i < numTensors; i++)
    {
        const char* name = mEngine->getIOTensorName(i);
        if (!name) throw std::runtime_error("TensorRT returned an unnamed I/O tensor");
        mTensorNames[i] = name;
        nvinfer1::Dims dims = mContext->getTensorShape(name);
        size_t elementCount = volume(dims);
        mHostBuffers[i].resize(elementCount);
        if (cudaMalloc(&mDeviceBuffers[i], elementCount * sizeof(float)) != cudaSuccess) 
        {
            throw std::runtime_error("CUDA memory allocation failed");
        }
        if (!mContext->setTensorAddress(name, mDeviceBuffers[i]))
            throw std::runtime_error("Failed to bind TensorRT tensor: " + mTensorNames[i]);

        if (mTensorNames[i] == "rgb") mRgbIndex = i;
        else if (mTensorNames[i] == "depth") mDepthIndex = i;
        else if (mTensorNames[i] == "mask") mMaskIndex = i;
        else if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
            mOutputIndex = i;
    }

    if (mRgbIndex < 0 || mDepthIndex < 0 || mMaskIndex < 0 || mOutputIndex < 0)
        throw std::runtime_error("TensorRT engine must expose rgb, depth, mask and one output");
}

size_t DepthCompleter::volume(const nvinfer1::Dims& dims) 
{
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; i++) v *= dims.d[i];
    return v;
}

void DepthCompleter::prepareInputs(const cv::Mat& rgbImage, const cv::Mat& depthImage) 
{
    // RGB (HWC -> CHW)
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(rgbImage, rgbChannels);
    for (int c = 0; c < 3; c++) 
    {
        std::memcpy(mHostBuffers[mRgbIndex].data() + c * mInputHeight * mInputWidth,
                    rgbChannels[c].data,
                    mInputHeight * mInputWidth * sizeof(float));
    }

    // Depth
    std::memcpy(mHostBuffers[mDepthIndex].data(), depthImage.data, mInputHeight * mInputWidth * sizeof(float));

    // Mask
    cv::Mat mask = depthImage > 0;  // CV_8U  0｜255
    mask.convertTo(mask, CV_32F, 1.0/255.0);
    std::memcpy(mHostBuffers[mMaskIndex].data(), mask.data, mInputHeight * mInputWidth * sizeof(float));

    // Copy to device
    for (int i : {mRgbIndex, mDepthIndex, mMaskIndex})
    {
        if (cudaMemcpy(mDeviceBuffers[i], mHostBuffers[i].data(), 
                       mHostBuffers[i].size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) 
        {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }
}

cv::Mat DepthCompleter::processOutput() 
{
    if (cudaMemcpy(mHostBuffers[mOutputIndex].data(), mDeviceBuffers[mOutputIndex],
                   mHostBuffers[mOutputIndex].size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) 
    {
        throw std::runtime_error("CUDA memcpy failed");
    }
    
    cv::Mat result(mInputHeight, mInputWidth, CV_32F, mHostBuffers[mOutputIndex].data());

    return result * 200.0f;
}
