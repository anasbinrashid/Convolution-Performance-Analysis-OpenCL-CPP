# Parallel Image Convolution Processing

A performance comparison study between CPU-based scalar processing and GPU-accelerated OpenCL implementation for image convolution operations using Sobel edge detection filters.

## Overview

This project implements and compares two approaches for applying convolution filters to images:
- **CPU Implementation**: Traditional scalar processing using OpenCV
- **GPU Implementation**: Parallel processing using OpenCL kernels

The assignment demonstrates the performance benefits of parallel computing by applying a 3x3 Sobel filter to a dataset of images and measuring execution times.

## Dependencies

### Required Libraries
- **OpenCV 4.x**: Image processing and I/O operations
- **OpenCL**: GPU computing framework
- **C++17**: Standard library features (filesystem)

### System Requirements
- OpenCL compatible GPU (NVIDIA, AMD, or Intel)
- OpenCL drivers installed
- CMake 3.10 or higher
- GCC/Clang with C++17 support

## Installation

### Ubuntu/Debian
```bash
# Install OpenCV
sudo apt update
sudo apt install libopencv-dev

# Install OpenCL
sudo apt install opencl-headers ocl-icd-opencl-dev

# For NVIDIA GPUs
sudo apt install nvidia-opencl-dev

# For AMD GPUs
sudo apt install mesa-opencl-icd
```

### Building the Project
```bash
git clone https://github.com/yourusername/parallel-image-convolution.git
cd parallel-image-convolution

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Or compile directly
g++ -std=c++17 -o scalar ../src/Scalar.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -o opencl ../src/OpenCl.cpp `pkg-config --cflags --libs opencv4` -lOpenCL
```

## Algorithm Details

### Sobel Edge Detection Filter
The implementation uses a 3x3 Sobel filter for vertical edge detection:
```
[ 1  0 -1 ]
[ 1  0 -1 ]
[ 1  0 -1 ]
```

### CPU Implementation Features
- Sequential pixel-by-pixel processing
- OpenCV's `copyMakeBorder` for edge handling
- Automatic normalization to 0-255 range
- Timing measurement using `chrono`

### GPU Implementation Features
- Parallel kernel execution using OpenCL
- 2D work-group distribution (width × height)
- Vectorized operations with `float4`
- Boundary checking within kernel
- Memory-efficient buffer management

## Performance Analysis

### Key Metrics
- **Processing Time**: Per-image execution time
- **Throughput**: Images processed per second
- **Speedup**: GPU time vs CPU time ratio
- **Memory Usage**: Buffer allocation efficiency

### Expected Results
- GPU implementation typically shows 10-50x speedup for large images
- Performance gain increases with image size
- Memory transfer overhead affects small images

## Technical Implementation

### OpenCL Kernel Highlights
```c
__kernel void convolution(__global const float* input, 
                         __global float* output, 
                         __constant float* filter, 
                         int width, int height, int filtersize)
```
- Global memory access patterns optimized
- Boundary condition handling
- Vectorized computation using `float4`

### Memory Management
- **Input Buffer**: Read-only image data
- **Output Buffer**: Write-only result data  
- **Filter Buffer**: Constant kernel coefficients
- Efficient host-device memory transfers

## Troubleshooting

### Common Issues

**OpenCL Platform Not Found**
```bash
# Check available platforms
clinfo

# Install OpenCL runtime
sudo apt install ocl-icd-libopencl1
```

**OpenCV Not Found**
```bash
# Verify installation
pkg-config --modversion opencv4

# If not found, install development packages
sudo apt install libopencv-contrib-dev
```

**Compilation Errors**
- Ensure C++17 support: `-std=c++17`
- Link OpenCL library: `-lOpenCL`
- Include OpenCV flags: `pkg-config --cflags --libs opencv4`

## Extending the Project

### Possible Enhancements
1. **Additional Filters**: Gaussian blur, Laplacian, custom kernels
2. **Color Images**: RGB channel processing
3. **Memory Optimization**: Shared memory usage in kernels
4. **Batch Processing**: Multiple images simultaneously
5. **Performance Profiling**: Detailed timing analysis
6. **Multi-GPU Support**: Distributed processing

### Research Directions
- Compare with CUDA implementation
- Analyze memory bandwidth utilization
- Study work-group size optimization
- Investigate different OpenCL devices

## Contributing

This is an academic project, but suggestions and improvements are welcome:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
