#include <omp.h>
#include <iostream>
using namespace std;

int main() {
    const int n = 10000000;
    int* arr = new int[n];
    for (int i = 0; i < n; i++) arr[i] = i + 1;

    long long sum_seq = 0;
    double start_seq = omp_get_wtime();

    // Sequential Sum
    for (int i = 0; i < n; i++) {
        sum_seq += arr[i];
    }

    double end_seq = omp_get_wtime();

    // Parallel Sum
    long long sum_par = 0;
    double start_par = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum_par)
    for (int i = 0; i < n; i++) {
        sum_par += arr[i];
    }

    double end_par = omp_get_wtime();

    cout << "Sequential Sum: " << sum_seq << ", Time: " << (end_seq - start_seq) << " sec" << endl;
    cout << "Parallel Sum:   " << sum_par << ", Time: " << (end_par - start_par) << " sec" << endl;

    delete[] arr;
    return 0;
}
