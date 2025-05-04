#include<bits/stdc++.h>

using namespace std;




int main(){

    int n;
    cin>>n;
vector<int>v(n);
for(int i=0;i<n;i++)cin>>v[i];
for(int i=0;i<n-1;i++){
 #pragma omp parallel for    
   for(int j=1;j<n-1-i;j++){
    if(v[j]>v[j+1]){
        swap(v[j+1],v[j]);
    }
   }
}
for(int i=0;i<n-1;i++){
 #pragma omp parallel for    
   for(int j=0;j<n-1-i;j++){
    if(v[j]>v[j+1]){
        swap(v[j+1],v[j]);
    }
   }
}
for(int i=0;i<n;i++)cout<<v[i]<<" ";


}