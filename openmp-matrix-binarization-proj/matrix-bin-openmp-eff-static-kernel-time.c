#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef unsigned char uint8_t;

static inline void expand_matrix(int* M, int dim){
    int S = dim + 2;
    for (int i = 1; i <= dim; i++) {
        M[i*S] = M[i*S + 1];                    //copia colonna 1 in colonna 0
        M[i*S + (dim+1)] = M[i*S + dim];        //copia colonna dim in colonna dim+1
        M[i] = M[S + i];                        //copia riga 1 in riga 0
        M[(dim+1)*S + i] = M[dim*S + i];        //copia riga dim in riga dim+1
    }
    M[0] = M[S + 1];                            //angolo in alto a sinista
    M[dim+1] = M[S + dim];                      //angolo in alto a destra
    M[(dim+1)*S] = M[dim*S + 1];                //angolo in basso a sinistra
    M[(dim+1)*S + (dim+1)] = M[dim*S + dim];    //angolo in basso a destra
}

int main(int argc, char** argv){
    if(argc != 3){
        fprintf(stderr,"Usage error.\nCorrect usage: %s NUM_OF_THREADS MATRIX_DIM\nNUM_OF_THREADS represents the number of threads computing the final matrix.\nMATRIX_DIM represents the number of rows (or cols) of the (MATRIX_DIM, MATRIX_DIM) matrixes (must be >=2000).", argv[0]);
        exit(EXIT_FAILURE);
    }
    int i, j, previous_row, my_row, next_row;
    double start, end;
    int threads_num = atoi(argv[1]);
    int N = atoi(argv[2]);
    if(N < 2000){
        fprintf(stderr,"Usage error.\nCorrect usage: %s NUM_OF_THREADS MATRIX_DIM\nNUM_OF_THREADS represents the number of threads computing the final matrix.\nMATRIX_DIM represents the number of rows (or cols) of the (MATRIX_DIM, MATRIX_DIM) matrixes (must be >=2000).", argv[0]);
        exit(EXIT_FAILURE);
    }
    int* A = (int*) malloc(sizeof(int)*(N+2)*(N+2));    //matrici allocate come array per avere la maggior efficienza possibile
    uint8_t* T = (uint8_t*) malloc(sizeof(uint8_t)*N*N); // T di uint8_t per meno memoria tanto contiene solo 0 e 1
    srand((unsigned int)time(NULL));
    for(i = 1; i < N+1; i++){
        for(j = 1; j < N+1; j++){
            A[i*(N+2)+j] = rand() % 100;
        }
    }
    expand_matrix(A, N);
    start = omp_get_wtime();    //inizio misurazione per tempo di esecuzione
    #pragma omp parallel num_threads(threads_num) shared(A,T,N) private(i,j,previous_row,my_row,next_row)
    {
        #pragma omp for schedule(static)
        for(i = 1; i <= N; i++){
            previous_row = (i-1)*(N+2);
            my_row = i*(N+2);
            next_row = (i+1)*(N+2);
            for(j = 1; j <= N; j++){
                T[(i-1)*N+(j-1)] = (9*A[my_row+j] > A[previous_row+j-1]+ A[previous_row+j]+ A[previous_row+j+1]+
                                             A[my_row+j-1]+       A[my_row+j]+       A[my_row+j+1]+
                                             A[next_row+j-1]+     A[next_row+j]+     A[next_row+j+1]);
            }
        }
    }
    end = omp_get_wtime();      //fine misurazione per tempo di esecuzione
    printf("OPENMP tempo di esecuzione solo kernel: %lf secondi\n", end-start);
    free(A);
    free(T);
}