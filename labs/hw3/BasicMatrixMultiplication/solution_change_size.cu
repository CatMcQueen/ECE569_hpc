#include <cuda_runtime.h>
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
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
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
  int dim_param;

cudaEvent_t astartEvent, astopEvent;
float aelapsedTime;
cudaEventCreate(&astartEvent);
cudaEventCreate(&astopEvent);

  args = wbArg_read(argc, argv);

 //ali  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;  // set to correct value
  numCColumns = numBColumns;  // set to correct value
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows*numCColumns*sizeof(float));  

 //ali  wbTime_stop(Generic, "Importing data and creating memory on host");

//ali  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
//ali  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
//ali  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  
 //ali  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here for A, B and C
  cudaMalloc((void **) &deviceA, numARows*numAColumns*sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows*numBColumns*sizeof(float));
  cudaMalloc((void **) &deviceC, numARows*numBColumns*sizeof(float));
  
 //ali wbTime_stop(GPU, "Allocating GPU memory.");

 //ali wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here for A and B
    cudaMemcpy(deviceA, hostA, numAColumns*numARows*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBColumns*numBRows*sizeof(float), cudaMemcpyHostToDevice);
  
 //ali wbTime_stop(GPU, "Copying input memory to the GPU.");

  
  
  // ali: this loop will sweep blocks 4x4, 8x8, 16x16 and 32x32
  dim_param=4;
  for(dim_param=4; dim_param<33; dim_param=dim_param*2) {
    dim3 DimBlock(dim_param,dim_param,1);
    //@@ Initialize the grid dimensions here
    // use dim3 structure for setting grid dimensions
	int dimx = (numCColumns - 1)/dim_param + 1;
	int dimy = (numCRows - 1)/dim_param + 1;
	dim3 DimGrid(dimx, dimy);  // need to fill in DimGrid() call
    //ali  wbTime_start(Compute, "Performing CUDA computation");
    cudaEventRecord(astartEvent, 0);
   //@@ Launch the GPU Kernel here

  matrixMultiply<<< DimBlock, DimGrid>>>(deviceA, deviceB, deviceC,
                                numARows, numAColumns,
                                numBRows, numBColumns,
                                numCRows, numCColumns);  
  cudaDeviceSynchronize();
  //cudaThreadSynchronize();

  cudaEventRecord(astopEvent, 0);
  cudaEventSynchronize(astopEvent);
  cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  printf("Total execution time (ms) %f for block size %d x %d matrix size of %d x %d and %d x %d\n",aelapsedTime,dim_param,dim_param,numARows,numAColumns,numBRows,numBColumns);

  //ali  wbTime_stop(Compute, "Performing CUDA computation");

  // wbLog(TRACE, "The block dimensions are ", dim_param, " x ", dim_param);
  //ali  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCColumns*numCRows*sizeof(float), cudaMemcpyDeviceToHost);
  
  //ali  wbTime_stop(Copy, "Copying output memory to the CPU");
  wbSolution(args, hostC, numCRows, numCColumns);
} /* end of block size sweep */
//ali  wbTime_start(GPU, "Freeing GPU Memory");

//@@ Free the GPU memory here
   cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);


//ali  wbTime_stop(GPU, "Freeing GPU Memory");
//ali  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
