#include "lodepng.h"  // lasciare .h

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void sfuma(unsigned char* image, unsigned char* image2, unsigned width, unsigned height) {
    for(unsigned i=0; i<width; i++) {
        for (unsigned j=0; j<height; j++) {
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
            sum1 = 0;
            sum2 = 0;
            sum3 = 0;
            sum4 = 0;
            cont = 0;
        }
    }
}

int main(int argc, char *argv[]) {

    const char* filename1 = "image.png";
    unsigned error1;
    unsigned char* image1 = 0;
    unsigned width, height;

    error1 = lodepng_decode32_file(&image1, &width, &height, filename1);
    if(error1) printf("error %u: %s\n", error1, lodepng_error_text(error1));

    unsigned char* image2 = malloc(width * height * 4);
    sfuma(image1, image2, width, height);

    const char* filename2 = "sfocata.png";

    unsigned error2 = lodepng_encode32_file(filename2, image2, width, height);
    if(error2) printf("error %u: %s\n", error2, lodepng_error_text(error2));

    free(image1);
    free(image2);

  return 0;
}