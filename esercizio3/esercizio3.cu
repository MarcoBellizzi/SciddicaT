#include <stdio.h>
#include <stdlib.h>

void multiplySequential(int* m, int* n, int* r, int mHeight, int commonDim, int nWidth) {
    for(int i=0; i<mHeight; i++) {
        for(int j=0; j<nWidth; j++) {

            int sum = 0;
            for(int k=0; k<commonDim; k++) {
                sum += m[i*commonDim + k] * n[k*nWidth + j];
            }

	    r[i*nWidth + j] = sum;
        }
    }
}

__global__
void multiplyStraightforward(int* m, int* n, int* r, int mHeight, int commonDim, int nWidth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<mHeight && j<nWidth) {
        int sum = 0;
        for(int k=0; k<commonDim; k++) {
            sum += m[i*commonDim + k] * n[k*nWidth + j];
        }

        r[i*nWidth + j] = sum;
    }
}

__global__
void multiplyTiled(int * m, int* n, int* r, int mHeight, int commonDim, int nWidth) {
	int tileWidth = 32;

	 __shared__ int mShared[32][32];
	 __shared__ int nShared[32][32];

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	int pValue = 0;

	for(int p=0; p<=commonDim/tileWidth; p++) {
		if(p*tileWidth + threadIdx.y < commonDim && row<mHeight)
			mShared[threadIdx.x][threadIdx.y] = m[row*commonDim + p*tileWidth + threadIdx.y];
		else mShared[threadIdx.x][threadIdx.y] = 0;
		
		if(p*tileWidth + threadIdx.x < commonDim && col<nWidth)
			nShared[threadIdx.x][threadIdx.y] = n[(p*tileWidth + threadIdx.x)*nWidth + col];
		else nShared[threadIdx.x][threadIdx.y] = 0;
		
		__syncthreads();
		
		for(int k=0; k<tileWidth; k++) {
			pValue += mShared[threadIdx.x][k] * nShared[k][threadIdx.y];
		}

		__syncthreads();

	}
	
	if(row<mHeight && col<nWidth)	
		r[row*nWidth + col] = pValue;
}


int main() {

    int mHeight = 2000;
    int commonDim = 500;
    int nWidth = 2000; 


    int* m;
    int* n;
    int* r;

    cudaMallocManaged(&m, mHeight * commonDim * sizeof(int));
    cudaMallocManaged(&n, commonDim * nWidth * sizeof(int));
    cudaMallocManaged(&r, mHeight * nWidth * sizeof(int));

    for(int i=0; i<mHeight; i++){
        for(int j=0; j<commonDim; j++) {
            m[commonDim*i + j] = 1;
        }
    }


    for(int i=0; i<commonDim; i++){
        for(int j=0; j<nWidth; j++) {
            n[nWidth*i + j] = 1;
        }
    }

    dim3 block_size(32, 32, 1);
    dim3 number_of_blocks(ceil(mHeight/32.0), ceil(commonDim/32.0), 1);

//    multiplySequential(m, n ,r, mHeight, commonDim, nWidth);
//    multiplyStraightforward<<<number_of_blocks, block_size>>>(m, n, r, mHeight, commonDim, nWidth);
    multiplyTiled<<<number_of_blocks, block_size>>>(m, n, r, mHeight, commonDim, nWidth);

    cudaDeviceSynchronize();
    
    return 0;
}

