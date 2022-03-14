#include <cuda.h>
#include <iostream>

__global__
void veccInit(float* x, float* y, int n) {
	for(int i = blockDim.x * blockIdx.x + threadIdx.x;
	   i<n;
	   i += blockDim.x * gridDim.x) {
		x[i] = 1;
		y[i] = 2; 
	}
}

__global__
void veccAddKernel(float* x, float* y, int n ){
	for(int i = blockDim.x * blockIdx.x + threadIdx.x;
	   i<n;
	   i += blockDim.x * gridDim.x) {
		y[i] = x[i] + y[i]; 
	}
}

int main() {
	int n = 1<<28;
	int size = n * sizeof(float);
	int block_size = 1024;
	int number_of_block = ceil(n/block_size);

	float *x, *y;
	cudaMallocManaged(&x, size);
	cudaMallocManaged(&y, size);

	veccInit<<<number_of_block, block_size>>>(x,y,n);

	veccAddKernel<<<number_of_block, block_size>>>(x, y, n);

	cudaDeviceSynchronize();

	bool ok = true;
	for(int i=0; i<n; i++) {
		if(y[i] != 3)
			ok = false;
	}

	if(ok) 
		std::cout<<"ok\n";
	else
		std::cout<<"no\n";

	cudaFree(x);
	cudaFree(y);

	return 0;
}

