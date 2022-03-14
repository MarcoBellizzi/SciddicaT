#include "lodepng.c"   // includere il .c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void sfuma(unsigned char* image, unsigned char* image2, unsigned width, unsigned height) {
    unsigned sum1 = 0;
    unsigned sum2 = 0;
    unsigned sum3 = 0;
    unsigned sum4 = 0;
    unsigned cont = 0;
    
    for(unsigned i=0; i<width; i++) {
        for (unsigned j=0; j<height; j++) {

            for(int k=-3; k<4; k++) {
                for(int w=-3; w<4; w++) {
                    if(i+k >=0 && i+k<width && j+w>=0 && j+w<height) {
                        sum1 += image[4 * width * (i+k) + 4 * (j+w) + 0];
                        sum2 += image[4 * width * (i+k) + 4 * (j+w) + 1];
                        sum3 += image[4 * width * (i+k) + 4 * (j+w) + 2];
                        sum4 += image[4 * width * (i+k) + 4 * (j+w) + 3];
                        cont++;
                    }
                }
            }
            image2[4 * width * i + 4 * j + 0] = sum1 / cont;
            image2[4 * width * i + 4 * j + 1] = sum2 / cont;
            image2[4 * width * i + 4 * j + 2] = sum3 / cont;
            image2[4 * width * i + 4 * j + 3] = sum4 / cont;
            
            sum1 = 0;
            sum2 = 0;
            sum3 = 0;
            sum4 = 0;
            cont = 0;
        }
    }
}

__global__
void sfumaInParallelo(unsigned char* image, unsigned char* image2, unsigned width, unsigned height) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<width && j<height) {
        unsigned sum1 = 0;
        unsigned sum2 = 0;
        unsigned sum3 = 0;
        unsigned sum4 = 0;
        unsigned cont = 0;
        for(int k=-3; k<4; k++) {
            for(int w=-3; w<4; w++) {
                if(i+k >=0 && i+k<width && j+w>=0 && j+w<height) {
                    sum1 += image[4 * width * (i+k) + 4 * (j+w) + 0];
                    sum2 += image[4 * width * (i+k) + 4 * (j+w) + 1];
                    sum3 += image[4 * width * (i+k) + 4 * (j+w) + 2];
                    sum4 += image[4 * width * (i+k) + 4 * (j+w) + 3];
                    cont++;
                }
            }
        }
        image2[4 * width * i + 4 * j + 0] = sum1 / cont;
        image2[4 * width * i + 4 * j + 1] = sum2 / cont;
        image2[4 * width * i + 4 * j + 2] = sum3 / cont;
        image2[4 * width * i + 4 * j + 3] = sum4 / cont;
    }
}

__global__
void prova() {

}

int main(int argc, char *argv[]) {
    
    unsigned char* image1 = 0;
    unsigned width, height;
    unsigned error = lodepng_decode32_file(&image1, &width, &height, "image.png");   // to retrive width and height
    free(image1);

    unsigned n = width * height * 4;
    dim3 block_size(32, 32, 1);
    dim3 number_of_blocks(ceil(width/32.0), ceil(height/32.0), 1);

    unsigned char* image2 = 0;
    cudaMallocManaged(&image2, n * sizeof(unsigned char));
    error = lodepng_decode32_file(&image2, &width, &height, "image.png");

    unsigned char* image3 = 0;
    cudaMallocManaged(&image3, n * sizeof(unsigned char));

   // sfuma(image2, image3, width, height);
    sfumaInParallelo<<<number_of_blocks, block_size>>>(image2, image3, width, height);
   // prova<<<1, 1>>>();
    
    cudaDeviceSynchronize();

    error = lodepng_encode32_file("sfocata.png", image3, width, height);

    printf("done!\n");

    cudaFree(image2);
    cudaFree(image3);

    return 0;
}