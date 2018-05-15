#ifndef COMMON_H
#define COMMON_H

#include "utils.h"

// For all architectures, constant mem is limited to 64 KB.
// Here we limit the filter size to 40x40 coefficients => 25.6KB
// If you know the max width of filters used in practice, it might be interesting to define it here
// since MAX_FILTER_WIDTH * MAX_FILTER_WIDTH * 4   elements are transfered at each transform scale
//
// There are two approaches for inversion :
//  - compute the inverse filters into the previous constant arrays, before W.inverse(). It is a little slower.
//  - pre-compute c_kern_inv_XX once for all... faster, but twice more memory is used

#define MAX_FILTER_WIDTH 40

#ifdef SEPARATE_COMPILATION
extern __constant__ DTYPE c_kern_L[MAX_FILTER_WIDTH];
extern __constant__ DTYPE c_kern_H[MAX_FILTER_WIDTH];
extern __constant__ DTYPE c_kern_IL[MAX_FILTER_WIDTH];
extern __constant__ DTYPE c_kern_IH[MAX_FILTER_WIDTH];

extern __constant__ DTYPE c_kern_LL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
extern __constant__ DTYPE c_kern_LH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
extern __constant__ DTYPE c_kern_HL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
extern __constant__ DTYPE c_kern_HH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
#else
__constant__ DTYPE c_kern_L[MAX_FILTER_WIDTH];
__constant__ DTYPE c_kern_H[MAX_FILTER_WIDTH];
__constant__ DTYPE c_kern_IL[MAX_FILTER_WIDTH];
__constant__ DTYPE c_kern_IH[MAX_FILTER_WIDTH];

__constant__ DTYPE c_kern_LL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
__constant__ DTYPE c_kern_LH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
__constant__ DTYPE c_kern_HL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
__constant__ DTYPE c_kern_HH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
#endif

__global__ void w_kern_soft_thresh(DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE beta, int Nr, int Nc);
void w_call_soft_thresh(DTYPE** d_coeffs, DTYPE beta, w_info winfos);

DTYPE** w_create_coeffs_buffer(w_info winfos);
void w_free_coeffs_buffer(DTYPE** coeffs, int nlevels);
void w_copy_coeffs_buffer(DTYPE** dst, DTYPE** src, w_info winfos);

#endif
