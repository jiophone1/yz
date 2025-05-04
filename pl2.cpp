#include <stdio.h>
#include <iostream>
#include <omp.h>
#include<bits/stdc++.h>
using namespace std;

int main() {
    const int n = 1000000;
   vector<int>a(n);
    int sum = 0;

    // Sequential part
    double sqst = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    double sqend = omp_get_wtime();
    int seqSum = sum;

    // Parallel part
    sum = 0;  // ⚠️ Reset sum before using reduction
    double plst = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }

    double plend = omp_get_wtime();

    cout << "Sequential Sum: " << seqSum << ", Time: " << sqend - sqst << " sec" << endl;
    cout << "Parallel   Sum: " << sum    << ", Time: " << plend - plst << " sec" << endl;

    return 0;
}
