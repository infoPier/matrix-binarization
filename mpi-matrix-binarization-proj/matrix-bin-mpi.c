#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv){

    if(argc != 2){
        fprintf(stderr,"Usage error.\nCorrect usage: %s MATRIX_DIM\nMATRIX_DIM represents the number of rows (or cols) of the (MATRIX_DIM, MATRIX_DIM) matrixes (must be >=2000).", argv[0]);
    }

    int my_rank, size;

    /*  MATRIX INITIALIZATION   */
    int i, j;
    double start, end;
    int N = atoi(argv[1]);
    if(N < 2000){
        fprintf(stderr,"Usage error.\nCorrect usage: %s MATRIX_DIM\nMATRIX_DIM represents the number of rows (or cols) of the (MATRIX_DIM, MATRIX_DIM) matrixes (must be >=2000).", argv[0]);
    }
    int* A = (int*) malloc(sizeof(int)*(N+2)*(N+2));
    int* T = (int*) malloc(sizeof(int)*N*N);
    srand((unsigned int)time(NULL));
    for(i = 1; i < N+1; i++){
        for(j = 1; j < N+1; j++){
            A[i*(N+2)+j] = rand() % 100;
        }
    }
    /*  MPI PART  */
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (my_rank == 0)
        printf("Tempo di esecuzione = %fs\n", end-start);

    MPI_Finalize();    

}