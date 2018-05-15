#ifndef SI_H
#define SI_H

#include "utils.h"

// Possible states of the Wavelet class.
// It prevents, for example, W.inverse() from being run twice (since W.d_coeffs[0] is modified)
typedef enum w_state {
    W_INIT,             // The class has just been initialized (coeffs not computed)
    W_FORWARD,          // W.forward() has just been performed (coeffs computed)
    W_INVERSE,          // W.inverse() has just been performed (d_image modified, coeffs modified !)
    W_THRESHOLD,        // The coefficients have been modified
    W_CREATION_ERROR,   // Error when creating the SparseInpainting instance
    W_FORWARD_ERROR,    // Error when computing the forward transform
    W_INVERSE_ERROR,    // Error when computing the inverse transform
    W_THRESHOLD_ERROR   // Error when thresholding the coefficients
} w_state;


class SparseInpainting {
  public:
    // Members
    // --------
    DTYPE* d_image;
    DTYPE** d_coeffs;       // Wavelet coefficients, on device
    DTYPE* d_tmp;           // Temporary device array (to avoid multiple malloc/free)

    char wname[128];        // Wavelet name
    w_info winfos;
    w_state state;


    // Operations
    // -----------
    // Default constructor
    SparseInpainting();
    // Constructor : SparseInpainting from image
    SparseInpainting(DTYPE* img_r, int Nr, int Nc, const char* wname, int levels);
    // Constructor: copy
    SparseInpainting(const SparseInpainting &W);// Pass by non-const reference ONLY if the function will modify the parameter and it is the intent to change the caller's copy of the data
    // Constructor : SparseInpainting from coeffs
    //~ SparseInpainting(DTYPE** d_thecoeffs, int Nr, int Nc, const char* wname, int levels, int do_cycle_spinning);
    // Destructor
    ~SparseInpainting();

    // Methods
    // -------
    void projc(DTYPE* Omega, DTYPE* y);
    void forward();
    void soft_threshold(DTYPE beta);
    void inverse(); 
    int get_image(DTYPE* img);
    int get_image_d(DTYPE* d_img_r);    
};


#endif

