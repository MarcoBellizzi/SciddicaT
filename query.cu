#include <iostream>

int main() {

	int dev_count;
	cudaDeviceProp dev_prop;
	cudaGetDeviceCount(&dev_count);
	
	for (int i = 0; i < dev_count; i++) {
		cudaGetDeviceProperties(&dev_prop, i);
		printf("Number of multiprocessors -> %d \n", dev_prop.multiProcessorCount);
		printf("Max thread per block      -> %d \n", dev_prop.maxThreadsPerBlock);
		printf("Max threads dim 0         -> %d \n", dev_prop.maxThreadsDim[0]);
		printf("Max threads dim 1         -> %d \n", dev_prop.maxThreadsDim[1]);
		printf("Max threads dim 2         -> %d \n", dev_prop.maxThreadsDim[2]);
		printf("Max grid side 0           -> %d   // 2'31 -1\n", dev_prop.maxGridSize[0]);
		printf("Max grid size 1           -> %d \n", dev_prop.maxGridSize[1]);
		printf("Max grid size 2           -> %d \n", dev_prop.maxGridSize[2]);
		printf("Warp size                 -> %d \n", dev_prop.warpSize);
	}
	
	return 0;
}
