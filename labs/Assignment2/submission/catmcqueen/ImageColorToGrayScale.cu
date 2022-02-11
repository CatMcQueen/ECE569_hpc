#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//@@ INSERT DEVICE CODE HERE

__global__ void ColorToGrayscale(float *inImg, float *outImg, int width, int height) {
        int idx, grayidx;
        int col = blockDim.x * blockIdx.x + threadIdx.x; // row of image
        int row  = blockDim.y * blockIdx.y + threadIdx.y; // col of image
        int numchannel = 3; // since it's RGB there are 3 channels

        // x = col and y = row
        if (col < width && row < height) {
                // each spot is 3 big (rgb) so get the number of spots
                grayidx = row * width + col;
                idx     = grayidx * numchannel; // and multiply by three to get current index
                // to calculate the beginning of the 3 for that pixel
                float r = inImg[idx];           //red
                float g = inImg[idx + 1];       //green
                float b = inImg[idx + 2];       //blue
                outImg[grayidx]  = (0.21*r + 0.71*g + 0.07*b); // now convert to grayscale
        }
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE

  // 256 = 16 * 16
  dim3 BlockDim(16,16);
  dim3 GridDim; // gridDim set belo

  // set grid to work in x & y to run rows and columns both
  GridDim.x = (imageWidth + BlockDim.x - 1) / BlockDim.x;
  GridDim.y = (imageHeight + BlockDim.y - 1) / BlockDim.y;

  // call the greyscale function
  ColorToGrayscale<<<GridDim, BlockDim>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
