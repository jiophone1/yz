#include <iostream>
#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int main()
{
    int mi(INT_MAX);
    const int n = 10000;
    vector<int> v(n);
    for (int i = 0; i < n; i++)
    {
        v[i] = rand() % 100 + 1;
    }

    float st = omp_get_wtime();
    for (int i = 0; i < n; i++)
    {
        if (mi > v[i])
        {
            mi = v[i];
        }
    }
    float en = omp_get_wtime();

    float st2 = omp_get_wtime();
#pragma omp parallel for reduction(min : mi)

    for (int i = 0; i < n; i++)
    {
        if (mi > v[i])
        {
            mi = v[i];
        }
    }
    float en2 = omp_get_wtime();
    cout << mi << " " << en - st << " " << en2 - st2;

    return 0;
}
