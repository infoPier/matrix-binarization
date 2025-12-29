#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef unsigned char uint8_t;

static inline void expand_matrix(int* M, int dim){ //espando solo le colonne, ci penserà ogni processo ad espandersi le righe di cui ha bisogno
    int S = dim + 2;
    for (int i = 1; i <= dim; i++) {
        M[i*S] = M[i*S + 1];                    //copia colonna 1 in colonna 0
        M[i*S + (dim+1)] = M[i*S + dim];        //copia colonna dim in colonna dim+1
    }
}

int main(int argc, char** argv){

    if(argc != 2){
        fprintf(stderr,"Usage error.\nCorrect usage: %s MATRIX_DIM\nMATRIX_DIM represents the number of rows (or cols) of the (MATRIX_DIM, MATRIX_DIM) matrixes (must be >=2000).", argv[0]);
        exit(EXIT_FAILURE);
    }

    int my_rank, size;

    /*  INIT MATRICE   */
    double start, end;
    int N = atoi(argv[1]);
    if(N < 2000){
        fprintf(stderr,"Usage error.\nCorrect usage: %s MATRIX_DIM\nMATRIX_DIM represents the number of rows (or cols) of the (MATRIX_DIM, MATRIX_DIM) matrixes (must be >=2000).", argv[0]);
        exit(EXIT_FAILURE);
    }
    int NP2 = N + 2;
    int* A = NULL;
    uint8_t* T = NULL;
    /*  MPI  */
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int x, y;
    if(my_rank == 0){
        A = (int*) malloc(sizeof(int)*(NP2)*(NP2));
        T = (uint8_t*) malloc(sizeof(uint8_t)*N*N); // T di uint8_t per meno memoria tanto contiene solo 0 e 1
        srand((unsigned int)time(NULL));
        for(x = 1; x < N+1; x++){
            for(y = 1; y < N+1; y++){
                A[x*(NP2)+y] = rand() % 100;
            }
        }
        expand_matrix(A, N);
    }

    int rows_per_proc = N / size;
    int extra_rows = N % size;
    int local_rows = rows_per_proc + (my_rank < extra_rows); // (my_rank < extra_rows) ritorna 1 se vero 0 se falso (C standard)
    int* local_A = (int*) malloc(sizeof(int)*(local_rows+2)*(NP2));
    unsigned char* local_T = (unsigned char*) malloc(sizeof(unsigned char)*local_rows*N);

    int* rowsnumber_per_proc_scatter = NULL;
    int* offset_per_proc_scatter = NULL; //offset rispetto alla matrice originale
    int* rowsnumber_per_proc_gather = NULL;
    int* offset_per_proc_gather = NULL;
    
    if(my_rank == 0){ // calcolo quante righe inviare a ciascun processo
        rowsnumber_per_proc_scatter = (int*) malloc(sizeof(int)*size);
        offset_per_proc_scatter = (int*) malloc(sizeof(int)*size);
        rowsnumber_per_proc_gather = (int*) malloc(sizeof(int)*size);
        offset_per_proc_gather = (int*) malloc(sizeof(int)*size);
        int offset_scatter = NP2;
        int offset_gather = 0;
        for(x = 0; x < size; x++){
            int rows = rows_per_proc + (x < extra_rows);
            rowsnumber_per_proc_scatter[x] = rows * (NP2);
            rowsnumber_per_proc_gather[x] = rows * N;
            offset_per_proc_scatter[x] = offset_scatter;
            offset_per_proc_gather[x] = offset_gather;
            offset_scatter += rows * (NP2);
            offset_gather += rows * N;
        }
    }

    // per mandare le righe ai vari processi (se avessi usato la scatter normale avrei dovuto assumere per forza extra_rows = 0)
    MPI_Scatterv(A, rowsnumber_per_proc_scatter, offset_per_proc_scatter, MPI_INT, &local_A[(NP2)], local_rows*(NP2), MPI_INT, 0, MPI_COMM_WORLD);

    // rank 0 → ghost sopra = copia della prima riga locale
    if (my_rank == 0) {
        for (x = 0; x < (NP2); x++)
            local_A[x] = local_A[(NP2) + x];
    }

    // rank size-1 → ghost sotto = copia dell’ultima riga locale
    if (my_rank == size - 1) {
        for (x = 0; x < (NP2); x++)
            local_A[(local_rows + 1) * (NP2) + x] = local_A[local_rows * (NP2) + x];
    }


    /*MANDO "verso l'alto" (dal processo rank al processo rank-1) E RICEVO "dal basso" (dal processo rank al processo rank+1)*/
    if (my_rank > 0) {
        MPI_Send(&local_A[NP2], NP2, MPI_INT, my_rank - 1, 0, MPI_COMM_WORLD);
    }
    if (my_rank < size - 1) {
        MPI_Recv(&local_A[(local_rows + 1) * (NP2)], NP2, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /*MANDO "verso il basso" (dal processo rank al processo rank+1) E RICEVO "dall'alto" (dal processo rank al processo rank-1)*/
    if (my_rank < size - 1) {
        MPI_Send(&local_A[local_rows * (NP2)], NP2, MPI_INT, my_rank + 1, 1, MPI_COMM_WORLD);
    }
    if (my_rank > 0) {
        MPI_Recv(&local_A[0], NP2, MPI_INT, my_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    int previous_row, my_row, next_row;
    MPI_Barrier(MPI_COMM_WORLD); 
    start = MPI_Wtime(); //inizio misurazione tempo
    for(x=1; x <= local_rows; x++){
        previous_row = (x-1)*(NP2);
        my_row = x*(NP2);
        next_row = (x+1)*(NP2);
        for(y=1; y <= N; y++){
            local_T[(x-1)*N+(y-1)] = (9*local_A[my_row+y] > local_A[previous_row+y-1]+ local_A[previous_row+y]+ local_A[previous_row+y+1]+
                                                            local_A[my_row+y-1]+       local_A[my_row+y]+       local_A[my_row+y+1]+
                                                            local_A[next_row+y-1]+     local_A[next_row+y]+     local_A[next_row+y+1]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime(); //fine misurazione tempo
    MPI_Gatherv(local_T, local_rows * N, MPI_UINT8_T, T, rowsnumber_per_proc_gather, offset_per_proc_gather, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    if (my_rank == 0){
        //pattern di output threads_num N time_in_seconds
        printf("%d;%d;%f;\n", size, N, end-start);
        free(rowsnumber_per_proc_scatter);
        free(offset_per_proc_scatter);
        free(rowsnumber_per_proc_gather);
        free(offset_per_proc_gather);
        free(A);
        free(T);
    }
    free(local_A);
    free(local_T);
    MPI_Finalize();   
}