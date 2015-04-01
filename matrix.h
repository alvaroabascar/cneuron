
#include <stdio.h>
#ifndef __MATRIX__
#define __MATRIX__


void mcopy(int rows, int cols, double m[][cols], double dest[][cols]);

void vcopy(int l, double v1[], double v2[]);

void mprint(int rows, int cols, double m[][cols]);

void mfprint(FILE *fp, int rows, int cols, double m[][cols]);

void vprint(int l, double arr[]);

void vprint_float(int l, float arr[]);

void vfprint(FILE *fp, int l, double arr[]);

void madd(int rows, int cols, double m1[][cols],
          double m2[][cols], double m3[][cols]);

void vadd(int l, double v1[], double v2[]);

void mprod(int rows, int cols, double m1[][cols],
           int rows2, int cols2, double m2[][cols2],
           double m3[][rows2]);

void transp(int rows, int cols, double m1[][cols], double m2[][rows]);

void scprod(int rows, int cols, double m1[][cols],
            double m2[][cols], double m3[][cols]);

int switchrows(int n, int m, double M[][m], int row1, int row2);

int switchcols(int n, int m, double M[][m], int col1, int col2);

void vsubstract(int n, float *out, float *first, float *sec);

void vscalarprod(int n, float *out, float *first, float *sec);

float vprod(int n, float *first, float *sec);

int max_index(int n, float array[n]);

#endif
