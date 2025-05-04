#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void vectorAdd2D(int *a, int *b, int *c, int n)
{
    int idx = threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    const int rows = 2;
    const int cols = 3;
    const int n = rows * cols;

    // Flattened input arrays (2D ko row-wise flatten kiya gaya)
    int h_a[n] = {1, 2, 3, 1, 2, 3};
    int h_b[n] = {2, 4, 0, 2, 3, 0};
    int h_c[n];

    // Device memory allocation
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));

    // Host to Device transfer
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel call
    vectorAdd2D<<<1, n>>>(d_a, d_b, d_c, n);

    // Device to Host result
    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Output: Flat format
    cout << "Flat Output: ";
    for (int i = 0; i < n; ++i)
    {
        cout << h_c[i] << " ";
    }
    cout << endl;

    // Output: 2D Format
    cout << "2D Format Output:\n";
    for (int i = 0; i < n; ++i)
    {
        cout << h_c[i] << " ";
        if ((i + 1) % cols == 0)
            cout << endl;
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
