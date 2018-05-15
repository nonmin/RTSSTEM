/// ****************************************************************************
///  Xin Li 
///  lix3@ornl.gov
/// ****************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <cuda.h>
#include "si.h"
#include "io.h"

char infile_y[1000], infile_mask[1000], infile_wave[1000], outfile[1000];
long long  Nr = -1, Nc = -1, iter = 10, nlevels = -1, lambda = 0.8;

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {

    long long i;
    if (argc < 7)
    {
        printf("-sparse_img: Input file of the sparse scanned image \n");
        printf("-sparse_mk: Input file of sparse scanning mask \n");
        printf("-wavelet: Specify wavelet type, example: db2 \n");
        printf("-output: Output file of reconstructed image \n");
        printf("-Nr: Numer of rows of image \n");
        printf("-Nr: Numer of columns of image \n");
        printf("-nlevels: level of wavelet, usually 2,3,4 \n");    
        printf("-iter: iterations, default is 10 \n");
        printf("-lambda: soft-thersholding value, default is 0.8 \n");
        return 0;
    }

	if ((i = ArgPos((char *)"-sparse_img", argc, argv)) > 0) strcpy(infile_y, argv[i + 1]);
	if ((i = ArgPos((char *)"-sparse_mk", argc, argv)) > 0) strcpy(infile_mask, argv[i + 1]);
    if ((i = ArgPos((char *)"-wavelet", argc, argv)) > 0) strcpy(infile_wave, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(outfile, argv[i + 1]);
	if ((i = ArgPos((char *)"-Nr", argc, argv)) > 0) Nr = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-Nc", argc, argv)) > 0) Nc = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-nlevels", argc, argv)) > 0) nlevels = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);

    float* y = read_dat_file_float(infile_y, Nr*Nc);
    float* img_r = read_dat_file_float(infile_y, Nr*Nc);
    
    if (y == NULL) {
        puts("Error: could not load sparse image");
        exit(1);
    }

    float* Omega = read_dat_file_float(infile_mask, Nr*Nc);
    if (Omega == NULL) {
        puts("Error: could not load scanning mask");
        exit(1);
    }

    DTYPE* d_img_r;
    DTYPE* d_y;
    DTYPE* d_Omega;

    // y
    DTYPE* d_imgr_in;
    cudaMalloc(&d_imgr_in, Nr*Nc*sizeof(DTYPE));
    if (!y) cudaMemset(d_imgr_in, 0, Nr*Nc*sizeof(DTYPE));
    else {
        cudaMemcpyKind transfer;
        transfer = cudaMemcpyHostToDevice;
        cudaMemcpy(d_imgr_in, img_r, Nr*Nc*sizeof(DTYPE), transfer);
    }
    d_img_r = d_imgr_in;

    // y
    DTYPE* d_y_in;
    cudaMalloc(&d_y_in, Nr*Nc*sizeof(DTYPE));
    if (!y) cudaMemset(d_y_in, 0, Nr*Nc*sizeof(DTYPE));
    else {
        cudaMemcpyKind transfer;
        transfer = cudaMemcpyHostToDevice;
        cudaMemcpy(d_y_in, y, Nr*Nc*sizeof(DTYPE), transfer);
    }
    d_y = d_y_in;

    // Omega
    DTYPE* d_Omega_in;
    cudaMalloc(&d_Omega_in, Nr*Nc*sizeof(DTYPE));
    if (!Omega) cudaMemset(d_Omega_in, 0, Nr*Nc*sizeof(DTYPE));
    else {
        cudaMemcpyKind transfer;
        transfer = cudaMemcpyHostToDevice;
        cudaMemcpy(d_Omega_in, Omega, Nr*Nc*sizeof(DTYPE), transfer);
    }
    d_Omega = d_Omega_in;


    SparseInpainting SI(d_img_r, Nr, Nc, infile_wave, nlevels);
    const clock_t begin_time = clock();
    for (int k = 1; k <= iter; k++){

        SI.projc(d_Omega, d_y);
        
        SI.forward();

        SI.soft_threshold(3.2);

        SI.inverse();
        SI.get_image_d(d_img_r);

        SparseInpainting SI(d_img_r, Nr, Nc, infile_wave, nlevels);

    }
    SI.get_image(img_r);
    std::cout << "duration: " << float( clock () - begin_time )/ CLOCKS_PER_SEC <<"seconds \n";

    save(outfile, Nr, Nc, img_r);
    return 0;

}
