#ifndef IO_H
#define IO_H

float* read_dat_file_float(const char* fname, int len);
void save(char *outfile, int nr, int nc, float* arr);

#endif
