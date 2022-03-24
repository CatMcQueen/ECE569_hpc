// version 0
// global memory only interleaved version
// include comments describing your approach

#define MAX_BINS 127

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
		if (position >= 0 && x < num_elements) {
			atomicAdd(&(bins[position]),1);
		}
		i += stride; // increase i by stride until greater than numelements
	}
}


// version 1
// shared memory privatized version
// include comments describing your approach
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
	// setup shared histogram
	__shared__ unsigned int histo_private[num_bins];
	__syncthreads();

	// setup thread counters
	int x 		= threadIdx.x + blockIdx.x * blockDim.x;
	int stride 	= blockDim.x*gridDim.x;
	int tx 		= threadIdx.x;

	// fill private_histogram bins with 0s
	// do this with the striding block dim in x
	for (int j = tx; j < num_bins; j+= blockDim.x) {
		if ( j < num_bins) {
			histo_private[j] = 0.0; // fill it with zeros
		}
	}
	// wait until they're all set to 0 before continuing
	__syncthreads();

	// verify it's inside boundary
	while(x < num_elements) {
		position = input[x]; // get the position
		atomicAdd(&histo_private[position],1); // do the addition
		i += stride; // then change the stride 
	}
	// now wait for all the other threads
	__syncthreads();
	
	// write to global memory in the stries
	// from when we were writing 0s (ie stepping through elements using
	// block Dim as the strice
	for (int j = tx; j < num_threads; j += blockDim.x ) {
		if (j < num_threads) {
			atomicAdd(&bins[j], &histo_private[j]);
		}
	}
}


// version 2
// your method of optimization using shared memory 
// include DETAILED comments describing your approach
// for competition you need to include description of the idea
// where you borrowed the idea from, and how you implmented 
__global__ void histogram_shared_optimized(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {


}

// clipping function
// resets bins that have value larger than 127 to 127. 
// that is if bin[i]>127 then bin[i]=127

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
	int x  = threadIdx.x + blockIdx.x * blockDim.x;
	// check for if it's in the bounds
	if (x < num_bins) {
		if (bins[x] > MAX_BINS) { // then if it's over the threshold
			bins[x] = MAX_BINS; // set it to 127 
		}
	}
}
