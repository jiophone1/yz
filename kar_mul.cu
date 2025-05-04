#include <iostream>
#include <cstdlib> // for rand(), srand()
#include <ctime>   // for time()
using namespace std;

__global__ void matrixMul(const int *a, const int *b, int *c, int N)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N && col < N)
    {
        int sum = 0;
        for (int i = 0; i < N; i++)
        {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main()
{
    const int N = 12;    // You can change N to any size (e.g. 16, 64, 512, etc.)
    const int BLOCK = 4; // Block size (BLOCK x BLOCK threads per block)

    int h_a[N * N], h_b[N * N], h_c[N * N];

    srand(time(0));

    cout << "Matrix A:\n";
    for (int i = 0; i < N * N; i++)
    {
        h_a[i] = rand() % 10;
        cout << h_a[i] << " ";
        if ((i + 1) % N == 0)
            cout << endl;
    }

    cout << "\nMatrix B:\n";
    for (int i = 0; i < N * N; i++)
    {
        h_b[i] = rand() % 10;
        cout << h_b[i] << " ";
        if ((i + 1) % N == 0)
            cout << endl;
    }

    int *d_a, *d_b, *d_c;
    size_t size = N * N * sizeof(int);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 🧠 Grid and Block calculation using ceil formula
    dim3 blockSize(BLOCK, BLOCK);
    dim3 gridSize((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    matrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cout << "\nResult Matrix C (A x B):\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << h_c[i * N + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
