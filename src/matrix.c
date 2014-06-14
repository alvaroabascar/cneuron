#include <stdio.h>

/* mcopy: copies a matrix, given the number of rows and cols, the matrix (m2)
 * and a destination matrix (m1)*/
void mcopy(int rows, int cols, double m1[][cols], double m2[][cols])
{
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            m1[i][j] = m2[i][j];
        }
    }
}

/* vcopy: copies a vector "v2" of length "l" into the vector "v1" */
void vcopy(int l, double v1[], double v2[])
{
    while (-l > 0) {
        v1[l] = v2[l];
    }
}

/* mprint: prints a matrix, given the number of rows and cols, and the matrix */
void mprint(int rows, int cols, double m[][cols])
{
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)
            printf("%.10e%s", m[i][j], j < cols - 1 ? "  " : "");
        printf("\n");
    }
}

/* mprint: prints a matrix to a file stream *fp, given the number of rows
 * and cols, and the matrix */
void mfprint(FILE *fp, int rows, int cols, double m[][cols])
{
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)
            fprintf(fp, "%.10e%s", m[i][j], j < cols - 1 ? "  " : "");
        fprintf(fp, "\n");
    }
}

/* vprint: prints a vector, given its length n */
void vprint(int l, double arr[])
{
    int i;

    printf("[ ");
    for (i = 0; i < l; i++) {
        printf("%.5f%s", arr[i], i < l - 1 ? ", " : "");
    }
    printf(" ]\n");
}

/* vprint_float: prints a vector, given its length n */
void vprint_float(int l, float arr[])
{
    int i;

    printf("[ ");
    for (i = 0; i < l; i++) {
        printf("%.5f%s", arr[i], i < l - 1 ? ", " : "");
    }
    printf(" ]\n");
}

/* vfprint: prints a vector, given its length n, to a file stream *fp */
void vfprint(FILE *fp, int l, double arr[])
{
    int i;

    fprintf(fp, "[ ");
    for (i = 0; i < l; i++) {
        fprintf(fp, "%.15e%s", arr[i], i < l - 1 ? ", " : "");
    }
    fprintf(fp, " ]\n");
}

/* addm: given a number of rows and cols and three matrices, sums the first
   two matrices and saves the result in the third one */
void madd(int rows, int cols, double m1[][cols],
        double m2[][cols], double outm[][cols])
{
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            outm[i][j] = m1[i][j] + m2[i][j];
}

/* vadd: given two vectors "v1" and "v2" of length "l", performs the sum
 * and stores it in "v1"
 */
void vadd(int l, double v1[], double v2[])
{
    while (l-- > 0) {
        v1[l] += v2[l];
    }
}


/* mprod: given two matrices m1 and m2, with sizes
          r1xc1 and r2xc2 respecively, builds the product and
          saves it to m3 */
void mprod(int r1, int c1, double m1[][c1],
           int r2, int c2, double m2[][c2],
           double m3[][c2])
{
    double sum, tmp[r1][c2];
    int i, j, k;

    /* calculate product */
    for (i = 0; i < r1; i++)
        for (j = 0; j < c1; j++) {
            sum = 0;
            for (k = j; k < c1; k++) {
                sum += m1[i][k] * m2[k][j];
            }
            tmp[i][j] = sum;
        }
    /* save product into output matrix */
    for (i = 0; i < r1; i++)
        for (j = 0; j < c2; j++)
            m3[i][j] = tmp[i][j];
}

/* transp: given a matrix m1 with r rows and c cols, saves the transpose
           of m1 into m2 */
void transp(int rows, int cols, double m1[][cols], double m2[][rows])
{
    int i, j;
    double tmp[rows][cols];
    /* build transpose */
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            tmp[i][j] = m1[j][i];
    /* copy transpose to output matrix. This allows the function
       to work in case m1 and m2 are the same matrix */
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            m2[j][i] = tmp[i][j];
}

/* scprod: given two matrices m1 and m2 of size rowsxcols, performs 
           the scalar product and saves the result into m3 */
void scprod(int rows, int cols, double m1[][cols],
            double m2[][cols], double m3[][cols])
{
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            m3[i][j] = m1[i][j] * m2[i][j];
}

/* switchrows: given an n-by-m matrix M, and two integers row1 and
 * row2, interchanges row "row1" and "row2" in M */
int switchrows(int n, int m, double M[][m], int row1, int row2)
{
    int i;
    double tmp;
    if (row1 > n-1 || row2 > n-1 || row1 < 0 || row2 < 0) {
        return -1;
    }
    for (i = 0; i < m; i++) {
        tmp = M[row1][i];
        M[row1][i] = M[row2][i];
        M[row2][i] = tmp;
    }
    return 0;
}

/* switchcols: given an n-by-m matrix M, and two integers col1 and
 * col2, interchanges row "col1" and "col2" in M */
int switchcols(int n, int m, double M[][m], int col1, int col2)
{
    int i;
    double tmp;
    if (col1 > n-1 || col2 > n-1 || col1 < 0 || col2 < 0) {
        return -1;
    }
    for (i = 0; i < m; i++) {
        tmp = M[i][col1];
        M[i][col1] = M[i][col2];
        M[i][col2] = tmp;
    }
    return 0;
}

/* vsubstract:
 *      given three pointers to memory containing an "array" of n floats,
 *      substracts the third array to the second and saves the result in
 *      the first.
 */
void vsubstract(int n, float *out, float *first, float *sec)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = first[i] - sec[i];
}

/* vscalarprod:
 *      given three pointers to memory containing an "array" of n floats,
 *      performs the scalar product of the second two and saves the result
 *      in the first one.
 */
void vscalarprod(int n, float *out, float *first, float *sec)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = first[i] * sec[i];
}

/* vprod:
 *      given two pointers to memory containing an "array" of n floats,
 *      performs the vector product of the second two and returns the result
 */
float vprod(int n, float *first, float *sec)
{
    int i;
    float result;
    for (i = result = 0; i < n; i++)
        result += first[i] * sec[i];
    return result;
}

/* max_index:
 *      return the index of the biggest element
 */
int max_index(int n, float array[n])
{
    int i, index;
    float max;
    for (max = 0; n >= 0; n--) {
        if (array[n] > max) {
            max = array[n];
            index = n;
        }
    }
    return index;
}






