#include <cuda.h>
#include <iostream>

__global__
void veccAddKernel(float* x, float* y, int n ){
	for(int i = blockDim.x * blockIdx.x + threadIdx.x; i<n; i += blockDim.x * gridDim.x) {
		y[i] = x[i] + y[i]; 
	}
}

int main() {
	int n = 1<<20;
	int size = n * sizeof(float);
	int block_size = 256;
	int number_of_block = ceil(n/block_size);

	float *x, *y;
	cudaMallocManaged(&x, size);
	cudaMallocManaged(&y, size);

	for(int i=0; i<n; i++) {
		x[i] = 1;
		y[i] = 2;
	}

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

