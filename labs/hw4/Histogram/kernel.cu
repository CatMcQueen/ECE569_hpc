// version 0
// global memory only interleaved version
// include comments describing your approach
#include <cuda_runtime.h>
#include <wb.h>

#define MAX_BINS 4096 
#define MAX_VAL 127



__global__ void histogram_global_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
	// thread map
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	// each thread has threadnumber of consecutive elements
	while (x < num_elements) {
		// get number of position
		int position = input[x];
		if (position >= 0 && position <= num_bins) {
			atomicAdd(&(bins[position]),1);
		}
		x += stride; // increase i by stride until greater than numelements
	}
}


// version 1
// shared memory privatized version
// include comments describing your approach
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
	// setup shared histogram
	__shared__ unsigned int histo_private[MAX_BINS];
	__syncthreads();

	// tx thread idx
	int tx 		= threadIdx.x;

	// fill private_histogram bins with 0s
	// do this with the striding block dim in x
 	for (int j=tx; j < num_bins; j += blockDim.x) {
		if ( tx <= MAX_BINS) {
			histo_private[j] = 0.0; // fill it with zeros
		}
	}

	// wait until they're all set to 0 before continuing
	__syncthreads();

        // setup thread counters
        int x           = threadIdx.x + blockIdx.x * blockDim.x;
        int stride      = blockDim.x*gridDim.x;


	// verify it's inside boundary
	while(x < num_elements) {
		int position = input[x]; // get the position
		atomicAdd(&histo_private[position],1); // do the addition
		x += stride;
	}
	// now wait for all the other threads
	__syncthreads();
	
	// write to global memory
	// you have to do this in strides as well 
	// use blockDim strides like in the set to 0
	for (int j = tx; j < num_bins; j+= blockDim.x) {
		if (j <= MAX_BINS) {
			atomicAdd(&bins[j], histo_private[j]);
		}
	}

	__syncthreads();
}

#define WARPSIZE 32

// version 2
// your method of optimization using shared memory 
// include DETAILED comments describing your approach
// for competition you need to include description of the idea
// where you borrowed the idea from, and how you implmented 
__global__ void histogram_shared_optimized(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int bd = blockDim.x;
	const int R = 1;
	// the +1 is for the padding
	__shared__ int histo_private[(MAX_BINS+1)*R];

	__syncthreads();
	// warp indexes
	const int warpid = (int)( tx / WARPSIZE);
	const int lane = tx % WARPSIZE;
	const int warps_block = bd / WARPSIZE;

	// offset to per-block sub histogram
	const int off_rep = (MAX_BINS + 1) * (tx % R);

	// set const for interleaved read access
	// to reduce the number of overlapping warps
	// account for the case where warp doesnt divide into number of elements
	const int elem_per_warp = (num_elements - 1)/warps_block + 1;
	const int begin = elem_per_warp * warpid + WARPSIZE * bx + lane;
	const int end = elem_per_warp * (warpid + 1);
	const int step= WARPSIZE * gridDim.x; 

	// Initialize
	for (int pos = tx; pos < (MAX_BINS+1) *R; pos += bd) {
		histo_private[pos] = 0;
	}

	// wait for all threads to complete
	__syncthreads();



	// Main loop
	for (int i = begin; i < end; i += step) {
		int pos = i < num_elements ? input[i] : 0;
		int inc = i < num_elements ? 1 : 0; // read the global mem
		atomicAdd(&histo_private[off_rep + d], inc); // vote in the shared memory
	}

	// wait for threads to end
	__syncthreads();

	//merge per_block sub histograms and write to global memory
	for (int pos = tx; pos < MAX_BINS; pos += bd) {
		int sum = 0;
		for(int base = 0; base < (MAX_BINS+1) *R; base += (MAX_BINS + 1)){
			sum += histo_private[base + pos];
		}		
		atomicAdd(bins+pos, sum);
	}	

	// wait for all threads to complete
	__syncthreads();

}

// clipping function
// resets bins that have value larger than 127 to 127. 
// that is if bin[i]>127 then bin[i]=127

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
	int x  = threadIdx.x + blockIdx.x * blockDim.x;
	// check for if it's in the bounds
	if (x < num_bins) {
		if (bins[x] > MAX_VAL) { // then if it's over the threshold
			bins[x] = MAX_VAL; // set it to 127 
		}
	}
}
