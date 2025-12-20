#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define mean(a1,a2,a3,a4,a5,a6,a7,a8,a9) ((float)(a1+a2+a3+a4+a5+a6+a7+a8+a9))/9.0

static inline void expand_matrix(int** M, int dim){
    int i;
    for(i = 1; i < dim+1; i++){
        M[0][i] = M[1][i];
        M[dim+1][i] = M[dim][i];
        M[i][0] = M[i][1];
        M[i][dim+1] = M[i][dim];
    }
    M[0][0] = M[2][2];
    M[0][dim+1] = M[2][dim-1];
    M[dim+1][0] = M[dim-1][2];
    M[dim+1][dim+1] = M[dim-1][dim-1];

}

static inline void print_matrix(int** M, int dim){
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim-1; j++){
            printf("\t%d, ", M[i][j]);
        }
        printf("%d\n",M[i][dim-1]);
    }
}

int main(int argc, char** argv){

    if(argc != 3){
        fprintf(stderr,"Usage error.\nCorrect usage: %s NUM_OF_THREADS MATRIX_DIM\nNUM_OF_THREADS represents the number of threads computing the final matrix.\nMATRIX_DIM represents the number of rows (or cols) of the (MATRIX_DIM, MATRIX_DIM) matrixes", argv[0]);
    }

    int i, j;

    int threads_num = atoi(argv[1]);
    int N = atoi(argv[2]);

    int** A = (int**) malloc(sizeof(int*)*(N+2));
    int** T = (int**) malloc(sizeof(int*)*N);

    for(i = 0; i < N+2; i++){
        A[i] = (int*) malloc(sizeof(int)*(N+2));
        if(i<N)
            T[i] = (int*) malloc(sizeof(int)*N);
    }

    srand((unsigned int)time(NULL));
    for(i = 1; i < N+1; i++){
        for(j = 1; j < N+1; j++){
            A[i][j] = rand() % 100;
        }
    }

    expand_matrix(A, N);

    printf("\n\nMATRIX A:\n");
    print_matrix(A,N+2);

    #pragma omp parallel num_threads(threads_num) shared(A,T,N) private(i,j)
    {
        #pragma omp for
        for(i = 1; i < N; i++){
            for(j = 1; j < N; j++){

                T[i][j] = (A[i][j] > mean(A[i-1][j-1], A[i-1][j], A[i-1][j+1],
                                          A[i][j-1],   A[i][j],   A[i][j+1],
                                          A[i+1][j-1], A[i+1][j], A[i+1][j+1])) ? 
                                  0
                                : 1;
            }
        }

    }

    printf("\n\nMATRIX T:\n");
    print_matrix(T,N);

    free(A);
    free(T);

}