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
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define ROWS 1024
#define COLS 1024

void sumMatrixOnHost(int *A, int *B, int *C)
{
    for (int i = 0; i < ROWS * COLS; i++)
        C[i] = A[i] + B[i];
}

__global__ void sumMatrixOnGPU2D(int *A, int *B, int *C)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < ROWS && col < COLS)
    {
        int index = row * COLS + col;
        C[index] = A[index] + B[index];
    }
}

int main()
{
    int size = ROWS * COLS;
    size_t bytes = size * sizeof(int);

    int *A = (int *)malloc(bytes);
    int *B = (int *)malloc(bytes);
    int *C = (int *)malloc(bytes);
    int *D = (int *)malloc(bytes);

    if (!A || !B || !C || !D)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    srand(10);

    for (int i = 0; i < size; i++)
    {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    clock_t start = clock();
    sumMatrixOnHost(A, B, C);
    clock_t end = clock();

    double cpuTime = 1000.0 * (end - start) / CLOCKS_PER_SEC;

    int *dA, *dB, *dC;

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    int configs[][2] = {{8, 8}, {16, 16}, {32, 8}, {32, 16}};
    float bestTime = 1e9f;
    int bestX = 0, bestY = 0;

    printf("Matrix size: %d x %d\n", ROWS, COLS);
    printf("CPU execution time: %.4f ms\n", cpuTime);

    for (int k = 0; k < 4; k++)
    {
        int bx = configs[k][0];
        int by = configs[k][1];

        dim3 block(bx, by);
        dim3 grid((COLS + bx - 1) / bx,
                  (ROWS + by - 1) / by);

        cudaEvent_t startGPU, stopGPU;
        cudaEventCreate(&startGPU);
        cudaEventCreate(&stopGPU);

        sumMatrixOnGPU2D<<<grid, block>>>(dA, dB, dC);
        cudaDeviceSynchronize();

        cudaEventRecord(startGPU);

        for (int r = 0; r < 10; r++)
            sumMatrixOnGPU2D<<<grid, block>>>(dA, dB, dC);

        cudaEventRecord(stopGPU);
        cudaEventSynchronize(stopGPU);

        float gpuTime;
        cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);
        gpuTime /= 10.0f;

        cudaMemcpy(D, dC, bytes, cudaMemcpyDeviceToHost);

        int correct = 1;
        for (int i = 0; i < size; i++)
        {
            if (D[i] != C[i])
            {
                correct = 0;
                break;
            }
        }

        printf("Block %d x %d, Grid %d x %d: %.4f ms - %s\n",
               bx, by, grid.x, grid.y, gpuTime,
               correct ? "PASS" : "FAIL");

        if (correct && gpuTime < bestTime)
        {
            bestTime = gpuTime;
            bestX = bx;
            bestY = by;
        }

        cudaEventDestroy(startGPU);
        cudaEventDestroy(stopGPU);
    }

    printf("Best block configuration: %d x %d\n", bestX, bestY);
    printf("Best GPU execution time: %.4f ms\n", bestTime);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);
    free(D);

    cudaDeviceReset();
    return 0;
}
```

## OUTPUT:
<img width="675" height="292" alt="image" src="https://github.com/user-attachments/assets/76b3231e-26d0-43d9-ac49-95ac932c0eaf" />


## RESULT:
The host took 0.010 seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in 0.0052 seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
