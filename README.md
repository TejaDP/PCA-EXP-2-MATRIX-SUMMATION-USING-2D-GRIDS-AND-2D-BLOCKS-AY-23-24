# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>AIM: matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition</h3>
<h3>ENTER YOUR NAME: TEJA D P</h3>
<h3>ENTER YOUR REGISTER NO:212225040464</h3>
<h3>EX. NO 2</h3>
<h3>DATE:26/10/2026</h3>
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```
%%writefile matrix.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("CUDA Error: %s\n", cudaGetErrorString(error)); \
        exit(1); \
    } \
}

double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

void initialData(int *ip, int size)
{
    for(int i = 0; i < size; i++)
    {
        ip[i] = rand() % 100;
    }
}

void sumMatrixOnHost(int *A, int *B, int *C, int nx, int ny)
{
    for(int iy = 0; iy < ny; iy++)
    {
        for(int ix = 0; ix < nx; ix++)
        {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

void checkResult(int *hostRef, int *gpuRef, int N)
{
    int match = 1;

    for(int i = 0; i < N; i++)
    {
        if(hostRef[i] != gpuRef[i])
        {
            match = 0;
            printf("host %d gpu %d\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if(match)
        printf("Arrays match.\n");
    else
        printf("Arrays do not match.\n");
}

__global__ void sumMatrixOnGPU2D(int *A, int *B, int *C, int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int idx = iy * NX + ix;

    if(ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    printf("Using Device: %s\n", deviceProp.name);

    int nx = 1 << 10;
    int ny = 1 << 10;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    printf("Matrix size: %d x %d\n", nx, ny);

    int *h_A, *h_B, *hostRef, *gpuRef;

    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRef = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);

    if(!h_A || !h_B || !hostRef || !gpuRef)
    {
        printf("Host memory allocation failed\n");
        return 1;
    }

    srand(0);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    double iStart = seconds();

    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    double iElaps = seconds() - iStart;

    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    int *d_MatA, *d_MatB, *d_MatC;

    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    int dimx = 16;
    int dimy = 16;

    dim3 block(dimx, dimy);

    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y
    );

    iStart = seconds();

    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    printf("sumMatrixOnGPU2D <<< (%d,%d), (%d,%d) >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nxy);

    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());

    return 0;
}
```

## OUTPUT:

<img width="1066" height="387" alt="Screenshot 2026-11-02 083336" src="https://github.com/user-attachments/assets/82839295-6dc3-4e4b-ba1e-4f60e49ca8f3" />


## RESULT:

The host took **0.002461 seconds** to complete its computation, while the GPU took **0.000190 seconds** to complete the computation. The GPU outperformed the CPU by using parallel processing with integer variables. The results were verified successfully, as both CPU and GPU outputs matched. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
