#include<bits/stdc++.h>
#include <ctime>
#include <omp.h>
#include <atomic>
using namespace  std;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

void BFS(vector<vector<int>>graph, int &node,vector<bool>& visited_bool){
    int n = graph.size();
    vector<atomic<bool>> visited(n);
    for(int i=0;i<n;i++) visited[i] = visited_bool[i];
    vector<int> current_level,next_level;
    current_level.push_back(node);
    visited[node] = true;

    const int num_thread = omp_get_max_threads();
    const int chunk = max(1,(int)current_level.size()/num_thread*4);

    while(!current_level.empty()){
        next_level.clear();
        #pragma omp parallel
        {
            vector<int> local_next;
            #pragma omp for schedule(dynamic,chunk) nowait
            for(int i=0;i<current_level.size();i++){
                int v = current_level[i];
                for(int nie:graph[v]){
                    bool was_visited = visited[nie].exchange(true);
                    if(!was_visited){
                        local_next.push_back(nie);
                    }
                }
            }
            #pragma omp critical
            next_level.insert(next_level.end(),local_next.begin(),local_next.end());
        }
        current_level.swap(next_level);
    }
    for(int i=0;i<visited_bool.size();i++){
        visited_bool[i] = visited[i];
    }    
}
void parallel_bfs(vector<vector<int>> graph,int start){
    int n = graph.size();
    vector<bool> visited_bool(n,false);
    BFS(graph,start,visited_bool);
}
void sequence_bfs(vector<vector<int>> graph,int start){
    int n = graph.size();
    vector<bool> visited(n,false);
    queue<int> container;
    container.push(start);
    visited[start] = 1;
    while(!container.empty()){
        int v = container.front();
        container.pop();
        for(int u:graph[v]){
            if(!visited[u]){
                visited[u] = true;
                container.push(u);
            }
        }
    }
}
int analysis(std::function<void()> function){
    auto start = high_resolution_clock::now();
    function();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);
    return duration.count();
}
int main(){
    vector<vector<int>> graph;;
    int n = 1000;
    int e = 500000;
    int speed_up = 0.0f;
    graph.resize(n);
    for(int i =0;i<e;i++){
        int u = (rand() % n);
        int v = (rand() % n);

        graph[u].push_back(v);
        graph[v].push_back(u);
    }
    int seq = analysis([&]{sequence_bfs(graph,0);});
    int par = analysis([&]{parallel_bfs(graph,0);});
    cout<<"seq"<<": "<<seq<<endl;
    cout<<"par"<<": "<<par<<endl;
    cout<<"diff"<<": "<<seq-par<<endl;

}