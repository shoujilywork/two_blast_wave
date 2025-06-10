// split_lf.h
#ifndef SPLIT_LF_H
#define SPLIT_LF_H

void split_lf(
    const double* r, const double* p, 
    const double* u, const double* v, const double* E,
    double* fp1, double* fp2, double* fp3, double* fp4,
    double* fn1, double* fn2, double* fn3, double* fn4,
    double* gp1, double* gp2, double* gp3, double* gp4,
    double* gn1, double* gn2, double* gn3, double* gn4,
    int rows, int cols);

#endif // SPLIT_LF_H