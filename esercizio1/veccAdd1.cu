#include <cuda.h>
#include <iostream>

__global__
void veccAddKernel(float* x, float* y, int n ){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n) { 
		y[i] += x[i]; 
	}
		
}

int main() {
	int n = 1<<20;
	int size = n * sizeof(float);
	dim3 block_size(256,1,1);
	dim3 number_of_block(ceil(n/block_size.x),1,1);

	float *x = new float[n];
	float *y = new float[n];
	
	float *X, *Y;

	for(int i=0; i<n; i++) {
		x[i] = 1;
		y[i] = 2;
	}

	cudaMalloc((void**)&X, size);
	cudaMalloc((void**)&Y, size);

	cudaMemcpy(X, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Y, y, size, cudaMemcpyHostToDevice);

	veccAddKernel<<<number_of_block, block_size>>>(X, Y, n);

	cudaMemcpy(y, Y, size, cudaMemcpyDeviceToHost);

	bool ok = true;
	for(int i=0; i<n; i++) {
		if(y[i] != 3)
			ok = false;
	}

	if(ok) 
		std::cout<<"ok\n";
	else
		std::cout<<"no\n";

	cudaFree(X); cudaFree(Y);

	return 0;
}

