#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"

float* read_dat_file_float(const char* fname, int len) {
    FILE* fid = fopen(fname, "rb");
    if (fid == NULL) {
        printf("ERROR in read_dat_file_float(): could not read %s\n", fname);
        return NULL;
    }
    float* out = (float*) calloc(len, sizeof(float));
    fread(out, len, sizeof(float), fid);
    fclose(fid);
    return out;
}


void save(char *outfile, int nr, int nc, float* arr)
{
    FILE *fout = fopen(outfile, "wb");
    for (long long i = 0; i < nr; ++i)
    {
        for (long long j = 0; j < nc; ++j)
        {   
            if (j) fprintf(fout, " ");
            fprintf(fout, "%f", arr[i * nc + j]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
}
