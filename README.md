# Matrix binarization
This repo contains the code for my project about matrix binarization related to the "Sistemi concorrenti e paralleli" course at Unibo.\
## What's matrix binarization
Give a matrix $A[N][N]$, the corrisponding binary matrix $T[N][N]$ is composed only by ones and zeros according to the following rules:
- for each $a_{ij}\in A$, $i\in[0,N-1]$ and $j\in[0,N-1]$ consider the 3x3 submatrix $A_{ij}[0..2][0..2]$
- having $m_{ij}$ representing the arithmetic mean of the values in the 3x3 submatrix $A_{ij}$

Every $t_{ij}\in T $, $i\in[0,N-1]$ and $j\in[0,N-1]$$ will have a value computed as follows:
- if $a_{ij}\gt m_{ij} \Longrightarrow t_{ij}=1$
- if $a_{ij}\le m_{ij} \Longrightarrow t_{ij}=0$

## Solutions
The project consists in providing the solution either in OPENMP and MPI.\
When both solutions are correctly running, a performance evaluation must be performed testing both strong and weak scalability.
