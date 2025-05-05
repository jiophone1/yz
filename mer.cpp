#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

// Merge two sorted parts
void mgsort(vector<int> &v, int st, int mi, int en)
{
    vector<int> tmp;
    int i = st, j = mi + 1;

    while (i <= mi && j <= en)
    {
        if (v[i] < v[j])
            tmp.push_back(v[i++]);
        else
            tmp.push_back(v[j++]);
    }
    while (i <= mi)
        tmp.push_back(v[i++]);
    while (j <= en)
        tmp.push_back(v[j++]);

    // Copy sorted part back to v
    for (int k = 0; k < tmp.size(); k++)
    {
        v[st + k] = tmp[k];
    }
}

// Sequential merge sort
void seq_merge_sort(vector<int> &v, int st, int en)
{
    if (st >= en)
        return;
    int mid = st + (en - st) / 2;
    seq_merge_sort(v, st, mid);
    seq_merge_sort(v, mid + 1, en);
    mgsort(v, st, mid, en);
}

// Parallel merge sort using OpenMP
void parallel_merge_sort(vector<int> &v, int st, int en)
{
    if (st >= en)
        return;
    int mid = st + (en - st) / 2;

#pragma omp parallel sections
    {
#pragma omp section
        parallel_merge_sort(v, st, mid);

#pragma omp section
        parallel_merge_sort(v, mid + 1, en);
    }

    mgsort(v, st, mid, en);
}

int main()
{
    const int n = 100000; // large input to show time difference
    vector<int> a(n), b(n);
    for (int i = 0; i < n; i++)
        a[i] = b[i] = rand() % 100000;

    double st1 = omp_get_wtime();
    seq_merge_sort(a, 0, n - 1);
    double en1 = omp_get_wtime();

    double st2 = omp_get_wtime();
    parallel_merge_sort(b, 0, n - 1);
    double en2 = omp_get_wtime();

    cout << "\nSequential Merge Sort Time: " << en1 - st1 << " seconds";
    cout << "\nParallel Merge Sort Time:   " << en2 - st2 << " seconds\n";

    // Uncomment to print sorted arrays if n is small
    // for (auto x : b) cout << x << " ";
    return 0;
}
