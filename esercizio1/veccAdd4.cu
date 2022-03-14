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

__global__
void check(float* y, bool* ok, int n) {
	for(int i = blockDim.x * blockIdx.x + threadIdx.x;
	   i<n;
	   i += blockDim.x * gridDim.x) {
		if(y[i] != 3)
			*ok = false; 
	}
}

int main() {
	int n = 1<<28;
	int size = n * sizeof(float);
	int block_size = 1024;
	int number_of_block = ceil(n/block_size);

	float *x, *y;
	bool *ok;
	cudaMallocManaged(&x, size);
	cudaMallocManaged(&y, size);
	cudaMallocManaged(&ok, sizeof(bool));

	*ok = true;

	veccInit<<<number_of_block, block_size>>>(x,y,n);

	veccAddKernel<<<number_of_block, block_size>>>(x, y, n);

	check<<<number_of_block, block_size>>>(y, ok, n);

	cudaDeviceSynchronize();

	if(ok) 
		std::cout<<"ok\n";
	else
		std::cout<<"no\n";

	cudaFree(x);
	cudaFree(y);
	cudaFree(ok);

	return 0;
}

