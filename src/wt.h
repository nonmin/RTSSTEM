#ifndef WT_H
#define WT_H
#include "utils.h"

int w_compute_filters_separable(const char* wname);

__global__ void kern_img_projc(DTYPE* d_image, DTYPE* Omega, DTYPE* y, int Nr, int Nc);
void img_projc(DTYPE* d_image, DTYPE* Omega, DTYPE* y, int Nr, int Nc);

__global__ void w_kern_forward_swt_pass1(DTYPE* img, DTYPE* tmp_a1, DTYPE* tmp_a2, int Nr, int Nc, int hlen, int level);
__global__ void w_kern_forward_swt_pass2(DTYPE* tmp_a1, DTYPE* tmp_a2, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level);
int w_forward_swt_separable(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);

__global__ void w_kern_inverse_swt_pass1(DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE* tmp1, DTYPE* tmp2, int Nr, int Nc, int hlen, int level);
__global__ void w_kern_inverse_swt_pass2(DTYPE* tmp1, DTYPE* tmp2, DTYPE* img, int Nr, int Nc, int hlen, int level);
int w_inverse_swt_separable(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);

#endif

