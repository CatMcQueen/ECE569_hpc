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

#define TILE_WIDTH 4 

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use tiling with shared memory for arbitrary size

  // create the tiles in shared memory
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

  // read in the variables 
  int tx = threadIdx.x;
  int ty = threadIdx.y; 
  int bx = blockIdx.x;
  int by = blockIdx.y;  
//blockDim.x * blockIdx.x + threadIdx.x;
  // get the row and column indexes to process  
  int row    = by * TILE_WIDTH + ty;
  int col    = bx * TILE_WIDTH + tx;

  float cval = 0;  

  // for each tile in the images
  for (int p = 0; p <(numAColumns+TILE_WIDTH -1)/TILE_WIDTH; ++p) {
    // check boundaries on A
    if ((p * TILE_WIDTH + tx) < numAColumns && row < numARows) {

    	// read in the right values for that tile and put them in the correct
    	// location in the shared memory
    	ds_A[ty][tx] = A[row*numAColumns+ p*TILE_WIDTH + tx];
    } else {
	ds_A[ty][tx] = 0.0;
    }
 
    // check boundaries again but for B
    if ((p*TILE_WIDTH + ty) < numBRows && col < numBColumns) {

       ds_B[ty][tx] = B[(p*TILE_WIDTH +ty) * numBColumns + col];  
     } else {
       ds_B[ty][tx] = 0.0;
     }   

    __syncthreads();


    // then process on the tile of data from shared memory for the partial product
    for (int i=0; i < TILE_WIDTH; i++) {
       cval += ds_A[ty][i] * ds_B[i][tx];
    }
    //std::cout << cval << std::endl;
    // set the correct memory location
    __syncthreads();
  }
  
  // then assign them back to C
  if (row < numCRows && col < numCColumns) {
     //int indexer = ((by * TILE_WIDTH + ty) * numCColumns) + (bx * TILE_WIDTH) + tx; //ty; // should that be tx?
      int indexer = row * numCColumns + col;
      C[indexer] = cval;
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
  int numCRows;    // number of rows in the matrix C(you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
                            
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;   // set to correct value
  numCColumns = numBColumns;   // set to correct value
  //@@ Allocate the hostC matrix

  hostC = (float *)malloc(numCRows*numCColumns*sizeof(float));

  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, numARows*numAColumns*sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows*numBColumns*sizeof(float));
  cudaMalloc((void **) &deviceC, numCRows*numCColumns*sizeof(float));


  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numAColumns*numARows*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBColumns*numBRows*sizeof(float), cudaMemcpyHostToDevice);  
  

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // note that TILE_WIDTH is set to 16 on line number 13. 
  dim3 BlockDim(TILE_WIDTH, TILE_WIDTH,1);
  //dim3 GridDim;
  
  // rows = y, column = x
  // do it in terms of C (outside edges of matrix multiply)
  // then add one to round up? maybe add .5?
  int dimx = (numCColumns - 1) / TILE_WIDTH + 1;
  int dimy = (numCRows - 1) / TILE_WIDTH+ 1;
  dim3 GridDim(dimx, dimy, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<GridDim,BlockDim>>>(deviceA, deviceB, deviceC,numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCColumns*numCRows*sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  wbTime_stop(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
