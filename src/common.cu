/// ****************************************************************************
/// ***************** Common utilities and  CUDA Kernels  **********************
/// ****************************************************************************

//~ #include "utils.h"
#include "common.h"
#define W_SIGN(a) ((a > 0) ? (1.0f) : (-1.0f))

/// soft thresholding of the detail coefficients (2D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
__global__ void w_kern_soft_thresh(DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_h[gidy*Nc + gidx];
        c_h[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);

        val = c_v[gidy*Nc + gidx];
        c_v[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);

        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }
}



/// ****************************************************************************
/// ******************** Common CUDA Kernels calls *****************************
/// ****************************************************************************

void  w_call_soft_thresh(DTYPE** d_coeffs, DTYPE beta, w_info winfos) {
    int tpb = 32; // Threads per block
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    int Nr = winfos.Nr, Nc = winfos.Nc, nlevels = winfos.nlevels;
    for (int i = 0; i < nlevels; i++) {
        n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
        w_kern_soft_thresh<<<n_blocks, n_threads_per_block>>>(d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], beta, Nr, Nc);
    }
}

/// Creates an allocated/padded device array : [ An, H1, V1, D1, ..., Hn, Vn, Dn]
DTYPE** w_create_coeffs_buffer(w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, nlevels = winfos.nlevels;

    DTYPE** res = (DTYPE**) calloc(3*nlevels+1, sizeof(DTYPE*));
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        cudaMalloc(&(res[i]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i], 0, Nr*Nc*sizeof(DTYPE));
        cudaMalloc(&(res[i+1]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i+1], 0, Nr*Nc*sizeof(DTYPE));
        cudaMalloc(&(res[i+2]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i+2], 0, Nr*Nc*sizeof(DTYPE));
    }
    // App coeff (last scale). They are also useful as a temp. buffer for the reconstruction, hence a bigger size
    cudaMalloc(&(res[0]), Nr*Nc*sizeof(DTYPE));
    cudaMemset(res[0], 0, Nr*Nc*sizeof(DTYPE));

    return res;
}

/// Deep free of wavelet coefficients
void w_free_coeffs_buffer(DTYPE** coeffs, int nlevels) {
    for (int i = 0; i < 3*nlevels+1; i++) cudaFree(coeffs[i]);
    free(coeffs);
}

/// Deep copy of wavelet coefficients. All structures must be allocated.
void w_copy_coeffs_buffer(DTYPE** dst, DTYPE** src, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, nlevels = winfos.nlevels;
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        cudaMemcpy(dst[i], src[i], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst[i+1], src[i+1], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst[i+2], src[i+2], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    // App coeff (last scale)
    cudaMemcpy(dst[0], src[0], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
}
