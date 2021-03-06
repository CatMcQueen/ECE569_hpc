
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
  //@@ Insert code to implement basic matrix multiplication for
  //@@ arbitrary size using global memory. 

  // calculate row of C and B, and the col of C and A
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;

  // make sure we can actually take the matrix multiply
  if (numAColumns != numBRows) {
     printf("Your matrix sizes do not work for a matrix multiply");
     return; 
  }

  // then the common width we're multiplying across is that common axis 
  //int width = numBRows;

  // if you're inside the matrix
  if ((row < numARows) && (col < numBColumns)) 
  {
    float Cval = 0;
    //each thread computes an element of the sub-matrix
    for (int k=0; k < numAColumns; k++) 
    {
      // get each element of the A row and the B col
      Cval += A[row*numAColumns + k] *  B[k*numBColumns+col];
    }
    //now save off the summation into the global memory of C
    C[row*numCColumns + col] = Cval;
  }

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

  // make sure we can actually take the matrix multiply
  if (numAColumns != numBRows) {
     printf("Your matrix sizes do not work for a matrix multiply");
     return;
  }

  FILE *f;
  f = fopen("basicMatrixMultiplication.log", "a+"); // Create a log file for printing
  fprintf(f, "Beginning execution ...... \n");

  args = wbArg_read(argc, argv);

  fprintf(f, "Made it past reading args\n");
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;   // set to correct value
  numCColumns = numBColumns;   // set to correct value
  //@@ Allocate the hostC matrix
 
  fprintf(f, "Made it past a and b\n");
  // allocate C memory
  hostC = (float *)malloc(numCRows*numCColumns*sizeof(float));
 
  fprintf(f, "Made it past allocating matrix\n");
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here for A, B and C
  cudaMalloc((void **) &deviceA, numARows*numAColumns*sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows*numBColumns*sizeof(float));
  cudaMalloc((void **) &deviceC, numARows*numBColumns*sizeof(float));

  fprintf(f, "Memory Allocated ...... /n");
  wbTime_stop(GPU, "Allocating GPU memory.");
  
  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here for A and B
  cudaMemcpy(deviceA, hostA, numAColumns*numARows*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBColumns*numBRows*sizeof(float), cudaMemcpyHostToDevice);
  
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // set block size to 16,16 and determine the grid dimensions
  // use dim3 structure for setting block and grid dimensions
  
  dim3 BlockDim(16,16,1);
  dim3 GridDim;

  // rows = y, column = x
  // do it in terms of C (outside edges of matrix multiply)
  // then add one to round up? maybe add .5?
  GridDim.x = (numCColumns - 1) / 16 + 1;
  GridDim.y = (numCRows - 1) / 16 + 1;

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
 fprintf(f, "Launching Kernel ...... /n");

  matrixMultiply<<< BlockDim, GridDim>>>(deviceA, deviceB, deviceC, 
                                numARows, numAColumns, 
                                numBRows, numBColumns,
                                numCRows, numCColumns);


  
  cudaDeviceSynchronize();
 fprintf(f, "Completed kernel  execution ...... /n");
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCColumns*numCRows*sizeof(float), cudaMemcpyDeviceToHost);

 fprintf(f, "Memory Copied ...... /n");
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
