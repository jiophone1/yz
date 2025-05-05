#include<bits/stdc++.h>
#include<ctime>
#include<omp.h>
#include<atomic>

using namespace std;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

void dfs(vector<vector<int>>& graph,int node, vector<bool> &visited_bool){
    int n = graph.size();
    vector<atomic<bool>> visited(n);
    for(int i=0;i<visited_bool.size();i++) visited[i] = visited_bool[i];

    visited[node].exchange(true);
    #pragma omp parallel
    #pragma omp single
    {
        for(int nie:graph[node]){
            if(!visited[nie].exchange(true));
            #pragma omp task firstprivate(nie)
            {
                stack<int> st;
                st.push(nie);
                while(!st.empty()){
                    int v  = st.top();
                    st.pop();
                    for(int w:graph[v]){
                        if(!visited[w].exchange(true)){
                            st.push(w);
                        }

                    }
                }
            }
        }
        #pragma omp taskwait
    }
    for(int i=0;i<visited.size();i++)visited_bool[i] = visited[i];
}
void DFS_without_threads(vector<vector<int>> &graph, vector<bool> & visited, int node) {
    visited[node] = true;

    for(int i = 0; i < graph[node].size(); i++){
        if(!visited[graph[node][i]])
            DFS_without_threads(graph, visited, graph[node][i]);
    }
}
void dfs_parallel(vector<vector<int>> graph,int start){
    int n = graph.size();
    vector<bool> visited(n);
    dfs(graph,start,visited);
}
void dfs(vector<vector<int>> graph,int start){
    int n = graph.size();
    vector<bool> visited(n);
    dfs(graph,start,visited);
}
int analysis(std::function<void()>function){
    auto start = high_resolution_clock::now();
    function();
    auto end = high_resolution_clock::now();
    auto diff = duration_cast<milliseconds>(end-start);
    return diff.count();
}
int main(){
    int n = 10000;
    int e = 500000;
    vector<vector<int>> graph;
    vector<bool> visited(n);
    graph.resize(n);
    for(int i=0;i<e;i++){
        int u = (rand() % n);
        int v = (rand() % n);
        graph[u].push_back(v);
        graph[v].push_back(u);
    }
    int seq = analysis([&]{DFS_without_threads(graph,visited,0);});
    int par = analysis([&]{dfs_parallel(graph,0);});
    float speed_up = seq*1.00/par;
    cout << "Sequential time : "<< seq << "ms" << endl;
    cout << "Parellel time : "<< par <<"ms"<<endl;
    cout << "Speed Up : "<<speed_up<<endl;
}

