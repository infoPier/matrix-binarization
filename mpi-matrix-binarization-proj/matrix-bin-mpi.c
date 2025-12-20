#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv){

    int my_rank, size;

    /*  MATRIX INITIALIZATION   */

    /*  MPI PART  */
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    

}