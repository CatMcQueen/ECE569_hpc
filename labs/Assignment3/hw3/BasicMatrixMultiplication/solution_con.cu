
// -*- c -*-
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

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement basic matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if matricies are of compatible size (redundant check from
  // current CPU code)
  if ( numARows != numBColumns ){
    return;
  }

  // Perform matrix multiplication
  if ( (row < numARows) && (col < numBColumns) ){
    float partialsum = 0; // Initialize the partial sum term
    for ( int j = 0; j < numAColumns; j++ ){
      partialsum += A[row * numAColumns + j] * B[j * numBColumns + col];
    }

    C[row * numCColumns + col] = partialsum;
  }

  // End kernel
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA; // A matrix on device
  float *deviceB; // B matrix on device
  float *deviceC; // C matrix on device
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  FILE *f;
  f = fopen("basicMatrixMultiplication.log", "a+"); // Create a log file for printing
  fprintf(f, "Beginning execution ...... \n");

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  fprintf(f, "Parsing input arguments and allocating A and B matricies ..... \n");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;   // set to correct value
  numCColumns = numBColumns;   // set to correct value
  //@@ Allocate the hostC matrix
  fprintf(f, "Allocating memory for matrix C\n");
  hostC = (float *)wbImport(wbArg_getInputFile(args, 2), &numCRows,
                            &numCColumns);

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  fprintf(f, "Allocate the GPU Memory for matricies A, B, and C ..... \n");
  //@@ Allocate GPU memory here for A, B and C
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float)); // Allocate device A on GPU
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float)); // Allocate device B on GPU
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float)); // Allocate device C on GPU

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  fprintf(f, "Copying A and B matricies to GPU ..... \n");
  //@@ Copy memory to the GPU here for A and B
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),
             cudaMemcpyHostToDevice); // Copy host matrix A to GPU
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
             cudaMemcpyHostToDevice); // Copy host matrix B to GPU
  // Do not need to copy matrix C to GPU

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // set block size to 16,16 and determine the grid dimensions
  // use dim3 structure for setting block and grid dimensions
  int blocksize = 16;
  dim3 blockDim (blocksize, blocksize, 1); // Setting to a 16x16 block dimension
  dim3 gridDim ((numCColumns - 1)/blocksize + 1, (numCRows - 1)/blocksize + 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  if (numAColumns != numBRows){
    fprintf(f, "ERROR!! Inner dimensions mismatch.\nExiting program.");
    cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceC);
    free(hostA); free(hostB); free(hostC);

    return -1;
  }
  fprintf(f, "Beginning CUDA computations ....... \n");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<blockDim, gridDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns,
                                      numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA); // Free matrix A
  cudaFree(deviceB); // Free matrix B
  cudaFree(deviceC); // Free matrix C

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

