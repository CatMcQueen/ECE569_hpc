#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
        //@@ Insert code to implement vector addition here
        //       and launch your kernel from the main function
        int tid;

        tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < len)
        {
                out[tid] = in1[tid] + in2[tid];

        }

//	for( int i=0; i < 5; i++) {
//		printf("%f and %f = %f \n", in1[tid], in2[tid], out[tid]);	
//	}
}

int main(int argc, char **argv) {
        wbArg_t args;
        int inputLength;
        float *hostInput1;
        float *hostInput2;
        float *hostOutput;
        float *deviceInput1;
        float *deviceInput2;
        float *deviceOutput;

        args = wbArg_read(argc, argv);

        wbTime_start(Generic, "Importing data and creating memory on host");
        hostInput1 =    (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
        hostInput2 =    (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
        hostOutput =    (float *)malloc(inputLength * sizeof(float));
        wbTime_stop(Generic, "Importing data and creating memory on host");

        wbLog(TRACE, "The input length is ", inputLength);

        wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here
        if(cudaMalloc((void **) &deviceInput1, inputLength*sizeof(float)) != cudaSuccess) {
                printf("malloc error on deviceInput1");
                return 0;
        }
        if(cudaMalloc((void **) &deviceInput2, inputLength*sizeof(float)) != cudaSuccess) {
                printf("malloc error on deviceInput2");
                cudaFree(deviceInput1);
                return 0;
        }

        if(cudaMalloc((void **) &deviceOutput, inputLength*sizeof(float)) != cudaSuccess) {
                printf("malloc error on deviceOutput");
                cudaFree(deviceInput1);
                cudaFree(deviceInput2);
                return 0;
        }


        wbTime_stop(GPU, "Allocating GPU memory.");

        wbTime_start(GPU, "Copying input memory to the GPU.");
        //@@ Copy memory to the GPU here
        if(cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
                printf("cudaMemcpy error, from host to device 1" );
                cudaFree(deviceInput1);
                cudaFree(deviceInput2);
                cudaFree(deviceOutput);
                return 0;
        }
        if(cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
                printf("cudaMemcpy error, from host to device 2" );
                cudaFree(deviceInput1);
                cudaFree(deviceInput2);
                cudaFree(deviceOutput);
                return 0;
        }
        if(cudaMemcpy(deviceOutput, hostOutput, inputLength*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
                printf("cudaMemcpy error, from host to device out");
                cudaFree(deviceInput1);
                cudaFree(deviceInput2);
                cudaFree(deviceOutput);
                return 0;
        }
        wbTime_stop(GPU, "Copying input memory to the GPU.");

        //@@ Initialize the grid and block dimensions here
	int blocksize = 1024; //  of threads per block
	int gridsize  = (int)ceil((float)inputLength/blocksize); // number of blocks in grid
        dim3 mygrid(gridsize);
        dim3 myblock(blocksize);

        wbTime_start(Compute, "Performing CUDA computation");
        //@@ Launch the GPU Kernel here
        vecAdd<<<mygrid, myblock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);


        cudaDeviceSynchronize();
        wbTime_stop(Compute, "Performing CUDA computation");

        wbTime_start(Copy, "Copying output memory to the CPU");
        //@@ Copy the GPU memory back to the CPU here

        if(cudaMemcpy(hostInput1, deviceInput1, inputLength*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                printf("cudaMemcpy error, from  device1 to host" );
                cudaFree(deviceInput1);
                cudaFree(deviceInput2);
                cudaFree(deviceOutput);
                return 0;
        }
        if(cudaMemcpy(hostInput2, deviceInput2, inputLength*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                printf("cudaMemcpy error, from device 2 to host" );
                cudaFree(deviceInput1);
                cudaFree(deviceInput2);
                cudaFree(deviceOutput);
                return 0;
        }
        if(cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                printf("cudaMemcpy error, from device out to host");
                cudaFree(deviceInput1);
                cudaFree(deviceInput2);
                cudaFree(deviceOutput);
                return 0;
        }

        wbTime_stop(Copy, "Copying output memory to the CPU");

        wbTime_start(GPU, "Freeing GPU Memory");
        //@@ Free the GPU memory here
	// release all 3 of the device memory blocks
        cudaFree(deviceInput1);
        cudaFree(deviceInput2);
        cudaFree(deviceOutput);

        wbTime_stop(GPU, "Freeing GPU Memory");

        wbSolution(args, hostOutput, inputLength);

        free(hostInput1);
        free(hostInput2);
        free(hostOutput);

        return 0;
}

