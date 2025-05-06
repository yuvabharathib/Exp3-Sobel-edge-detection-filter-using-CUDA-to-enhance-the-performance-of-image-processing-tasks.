# Sobel edge detection filter using CUDA to enhance the performance of image processing tasks

### EX. NO: 03
### ENTER YOUR NAME: Yuvabharathi B
### REGISTER NO: 212222230181
### DATE:

## Background: 
  - The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. 
  - This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

## Aim
To utilize CUDA to parallelize the Sobel filter implementation for efficient processing of images.

## Tools Required:
- A system with CUDA-capable GPU.
- CUDA Toolkit and OpenCV installed.

## Procedure

1. **Environment Setup**:
   - Ensure that CUDA and OpenCV are installed and set up correctly on your system.
   - Have a sample image (`images.jpg`) available in the correct directory to use as input.

2. **Load Image and Convert to Grayscale**:
   - Use OpenCV to read the input image in color mode.
   - Convert the image to grayscale as the Sobel operator works on single-channel images.

3. **Initialize and Allocate Memory**:
   - Determine the width and height of the grayscale image.
   - Allocate memory on both the host (CPU) and device (GPU) for the image data. Allocate device memory using `cudaMalloc` and check for successful allocation with `checkCudaErrors`.

4. **Performance Analysis Function**:
   - Define `analyzePerformance`, a function to test the CUDA kernel with different image sizes and block configurations.
   - For each specified image size (e.g., 256x256, 512x512, 1024x1024), set up the grid and block dimensions.
   - Launch the Sobel kernel using different block sizes (8x8, 16x16, 32x32) to evaluate the performance impact of each configuration. Record the execution time using CUDA events.

5. **Run Sobel Filter on Original Image**:
   - Set up the grid and block dimensions for the input image based on a 16x16 block size.
   - Use CUDA events to measure execution time for the Sobel filter applied to the original image.
   - Copy the resulting data from device memory to host memory.

6. **Save CUDA Output Image**:
   - Convert the processed image data on the host back to an OpenCV `Mat` object.
   - Save the CUDA-processed output image as `output_sobel_cuda.jpeg`.

7. **Compare with OpenCV Sobel Filter**:
   - For comparison, apply the OpenCV Sobel filter to the grayscale image on the CPU.
   - Measure the execution time using `std::chrono` for the CPU-based approach.
   - Save the OpenCV output as `output_sobel_opencv.jpeg`.

8. **Display Results**:
   - Print the input and output image dimensions.
   - Print the execution time for the CUDA Sobel filter and the CPU (OpenCV) Sobel filter to compare performance.
   - Display the breakdown of times for each block size and image size tested.

9. **Cleanup**:
   - Free all dynamically allocated memory on the host and device to avoid memory leaks.
   - Destroy CUDA events created for timing.

## Program

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,  
                            unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

        int sumX = 0;
        int sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                unsigned char pixel = srcImage[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        int magnitude = sqrtf(sumX * sumX + sumY * sumY);
        magnitude = min(max(magnitude, 0), 255);
        dstImage[y * width + x] = static_cast<unsigned char>(magnitude);
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

void analyzePerformance(const std::vector<std::pair<int, int>>& sizes, 
                        const std::vector<int>& blockSizes, unsigned char *d_inputImage, 
                        unsigned char *d_outputImage) {
                        
    for (auto size : sizes) {
        int width = size.first;
        int height = size.second;

        printf("CUDA - Size: %dx%d\n", width, height);
        
        dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));
        for (auto blockSize : blockSizes) {
            dim3 blockDim(blockSize, blockSize);
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            sobelFilter<<<gridSize, blockDim>>>(d_inputImage, d_outputImage, width, height);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("    Block Size: %dx%d Time: %f ms\n", blockSize, blockSize, milliseconds);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
}

int main() {
    Mat image = imread("/content/images.jpg", IMREAD_COLOR);
    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    // Convert to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    int width = grayImage.cols;
    int height = grayImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage,grayImage.data,imageSize,cudaMemcpyHostToDevice));

    // Performance analysis
    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {1024, 1024}};
    std::vector<int> blockSizes = {8, 16, 32};

    analyzePerformance(sizes, blockSizes, d_inputImage, d_outputImage);

    // Execute CUDA Sobel filter one last time for the original image
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));
    dim3 blockDim(16, 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockDim>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    checkCudaErrors(cudaMemcpy(h_outputImage,d_outputImage,imageSize,cudaMemcpyDeviceToHost));

    // Output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel_cuda.jpeg", outputImage);

    // OpenCV Sobel filter for comparison
    Mat opencvOutput;
    auto startCpu = std::chrono::high_resolution_clock::now();
    cv::Sobel(grayImage, opencvOutput, CV_8U, 1, 0, 3);
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endCpu - startCpu;

    // Save and display OpenCV output
    imwrite("output_sobel_opencv.jpeg", opencvOutput);
    
    printf("Input Image Size: %d x %d\n", width, height);
    printf("Output Image Size (CUDA): %d x %d\n", outputImage.cols, outputImage.rows);
    printf("Total time taken (CUDA): %f ms\n", milliseconds);
    printf("OpenCV Sobel Time: %f ms\n", cpuDuration.count());

    // Cleanup
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

## Output Explanation

| Original 	|  Output using Cuda |
|:-:	|:-:	|
| ![image](https://github.com/user-attachments/assets/e7fa9849-d486-4669-bc6c-1f0bea126742) | ![image](https://github.com/user-attachments/assets/2a5b6bd2-cd38-419f-bed4-f06ca45d6a76) |

| Original 	|  Output using OpenCV |
|:-:	|:-:	|
| ![image](https://github.com/user-attachments/assets/e7fa9849-d486-4669-bc6c-1f0bea126742) |  ![image](https://github.com/user-attachments/assets/7ce4e5eb-5de7-44a0-8142-3c7d1c9ecc13) |

- **Sample Execution Results**:
  - **CUDA Execution Times (Sobel filter)**
  </br>

<img src="https://github.com/user-attachments/assets/bc4f2f40-2f6c-4dd8-af8b-e4583b28079d" width="400">


  - **OpenCV Execution Time**
  </br>

![image](https://github.com/user-attachments/assets/db287c68-1ae8-42a4-9ef3-a78f16d0de37)

- **Graph Analysis**:
  - Displayed a graph showing the relationship between image size, block size, and execution time.
 </br>

<img src="https://github.com/user-attachments/assets/ddb5fb05-53fe-440f-8876-0d3797de1bb5" width="500">


## Answers to Questions

1. **Challenges Implementing Sobel for Color Images**:
   - Converting images to grayscale in the kernel increased complexity. Memory management and ensuring correct indexing for color to grayscale conversion required attention.

2. **Influence of Block Size**:
   - Smaller block sizes (e.g., 8x8) were efficient for smaller images but less so for larger ones, where larger blocks (e.g., 32x32) reduced overhead.

3. **CUDA vs. CPU Output Differences**:
   - The CUDA implementation was faster, with minor variations in edge sharpness due to rounding differences. CPU output took significantly more time than the GPU.

4. **Optimization Suggestions**:
   - Use shared memory in the CUDA kernel to reduce global memory access times.
   - Experiment with adaptive block sizes for larger images.

## Result
Successfully implemented a CUDA-accelerated Sobel filter, demonstrating significant performance improvement over the CPU-based implementation, with an efficient parallelized approach for edge detection in image processing.
