#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "common.h"
#include "si.h"
#include "wt.h"

#  define CUDACHECK \
  { cudaThreadSynchronize(); \
    cudaError_t last = cudaGetLastError();\
    if(last!=cudaSuccess) {\
      printf("ERRORX: %s  %s  %i \n", cudaGetErrorString( last),    __FILE__, __LINE__    );    \
      exit(1);\
    }\
  }


// FIXME: temp. workaround
#define MAX_FILTER_WIDTH 40


/// Constructor : default
SparseInpainting::SparseInpainting(void) : d_image(NULL), d_coeffs(NULL), d_tmp(NULL)
{
}


/// Constructor :  SparseInpainting from image
SparseInpainting::SparseInpainting(
    DTYPE* img_r,
    int Nr,
    int Nc,
    const char* wname,
    int levels) :

    d_image(NULL),
    d_coeffs(NULL),
    d_tmp(NULL),
    state(W_INIT)
{
    winfos.Nr = Nr;
    winfos.Nc = Nc;
    winfos.nlevels = levels;

    if (levels < 1) {
        puts("Warning: cannot initialize wavelet coefficients with nlevels < 1. Forcing nlevels = 1");
        winfos.nlevels = 1;
    }
    cudaMemcpyKind transfer;
    transfer = cudaMemcpyDeviceToDevice;

    // Image
    DTYPE* d_arr_in;
    cudaMalloc(&d_arr_in, Nr*Nc*sizeof(DTYPE));
    if (!img_r) cudaMemset(d_arr_in, 0, Nr*Nc*sizeof(DTYPE));
    cudaMemcpy(d_arr_in, img_r, Nr*Nc*sizeof(DTYPE), transfer);
    d_image = d_arr_in;

    DTYPE* d_tmp_new;
    cudaMalloc(&d_tmp_new, 2*Nr*Nc*sizeof(DTYPE)); // Two temp. images
    d_tmp = d_tmp_new;
    cudaMemset(d_tmp, 0, 2*Nr*Nc*sizeof(DTYPE));

    // Filters
    strncpy(this->wname, wname, 128);
    int hlen = 0;
    hlen = w_compute_filters_separable(wname);
    if (hlen == 0) {
        printf("ERROR: unknown wavelet name %s\n", wname);
        //~ exit(1);
        state = W_CREATION_ERROR;
    }
    winfos.hlen = hlen;

    // Compute max achievable level according to image dimensions and filter size
    int N;
    N = Nc;
    int wmaxlev = w_ilog2(N/hlen);
    // TODO: remove this limitation
    if (levels > wmaxlev) {
        printf("Warning: required level (%d) is greater than the maximum possible level for %s (%d) on a %dx%d image.\n", winfos.nlevels, wname, wmaxlev, winfos.Nc, winfos.Nr);
        printf("Forcing nlevels = %d\n", wmaxlev);
        winfos.nlevels = wmaxlev;
    }

    // Allocate coeffs
    DTYPE** d_coeffs_new;
    d_coeffs_new = w_create_coeffs_buffer(winfos);
    d_coeffs = d_coeffs_new;
}

/// Constructor: copy
SparseInpainting::SparseInpainting(const SparseInpainting &W) :
    state(W.state)
{
    winfos.Nr = W.winfos.Nr;
    winfos.Nc = W.winfos.Nc;
    winfos.nlevels = W.winfos.nlevels;
    winfos.hlen = W.winfos.hlen;

    strncpy(wname, W.wname, 128);
    cudaMalloc(&d_image, winfos.Nr*winfos.Nc*sizeof(DTYPE));
    cudaMemcpy(d_image, W.d_image, winfos.Nr*winfos.Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_tmp, 2*winfos.Nr*winfos.Nc*sizeof(DTYPE));

    d_coeffs = w_create_coeffs_buffer(winfos);
    w_copy_coeffs_buffer(d_coeffs, W.d_coeffs, winfos);

}

/// Destructor
SparseInpainting::~SparseInpainting(void) {
    if (d_image) cudaFree(d_image);
    if (d_coeffs) w_free_coeffs_buffer(d_coeffs, winfos.nlevels);
    if (d_tmp) cudaFree(d_tmp);
}

void SparseInpainting::projc(DTYPE* Omega, DTYPE* y) {

    img_projc(d_image, Omega, y, winfos.Nr, winfos.Nc);
}

/// Method : forward
void SparseInpainting::forward(void) {
    if (state == W_CREATION_ERROR) {
        puts("Warning: forward transform not computed, as there was an error when creating the wavelets");
        return;
    }
    w_forward_swt_separable(d_image, d_coeffs, d_tmp, winfos);
    // else: not implemented yet
    state = W_FORWARD;

}
/// Method : inverse
void SparseInpainting::inverse(void) {
    if (state == W_INVERSE) { // TODO: what to do in this case ? Force re-compute, or abort ?
        puts("Warning: W.inverse() has already been run. Inverse is available in W.get_image()");
        return;
    }
    if (state == W_FORWARD_ERROR || state == W_THRESHOLD_ERROR) {
        puts("Warning: inverse transform not computed, as there was an error in a previous stage");
        return;
    }
    w_inverse_swt_separable(d_image, d_coeffs, d_tmp, winfos);

    state = W_INVERSE;
}

/// Method : soft thresholding (L1 proximal)
void SparseInpainting::soft_threshold(DTYPE beta) {
    if (state == W_INVERSE) {
        puts("Warning: SparseInpainting(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_soft_thresh(d_coeffs, beta, winfos);
}

/// Method : get the image from device to host
int SparseInpainting::get_image(DTYPE* img_r) { // TODO: more defensive
    cudaMemcpy(img_r, d_image, winfos.Nr*winfos.Nc*sizeof(DTYPE), cudaMemcpyDeviceToHost);
    return winfos.Nr*winfos.Nc;
}

/// Method : get the image from device to device
int SparseInpainting::get_image_d(DTYPE* d_img_r) { // TODO: more defensive
    cudaMemcpy(d_img_r, d_image, winfos.Nr*winfos.Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    return winfos.Nr*winfos.Nc;
}







