#include <bits/stdc++.h> 
#include <ctime> 
#include <omp.h>
using namespace std;

using std :: chrono :: duration_cast;
using std :: chrono :: high_resolution_clock; 
using std :: chrono :: milliseconds;
void BFS(vector<vector<int>> &graph, int node, vector<bool>& visited_bool, bool &print_node) {
    int N = graph.size();
    // reinterpret your bool vector as atomic<bool>
    vector<atomic<bool>> visited(N);
    for(int i = 0; i < N; ++i) visited[i] = visited_bool[i];

    vector<int> current_level, next_level;
    current_level.push_back(node);
    visited[node] = true;

    const int num_threads = omp_get_max_threads();
    const int chunk = max(1, (int)current_level.size() / (num_threads * 4));

    while (!current_level.empty()) {
        next_level.clear();

        #pragma omp parallel
        {
            vector<int> local_next;
            #pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < (int)current_level.size(); ++i) {
                int v = current_level[i];

                for (int nei : graph[v]) {
                    // lock‐free test‐and‐set
                    bool was_visited = visited[nei].exchange(true);
                    if (!was_visited) {
                        local_next.push_back(nei);
                    }
                }
            }

            // merge once per thread
            #pragma omp critical
            next_level.insert(next_level.end(),
                              local_next.begin(),
                              local_next.end());
        }

        // prepare for next round
        current_level.swap(next_level);
    }

    // copy back for any further use
    for(int i = 0; i < N; ++i) visited_bool[i] = visited[i];
}
void parellel_BFS(vector<vector<int>>&graph, int start, bool &print_node){
    int N = graph.size();
    vector<bool> visited(N, false);
    BFS(graph, start, visited,print_node);
}

void sequential_BFS(vector<vector<int>> &graph, int start_node, bool& print_node){
    int N = graph.size();
    vector<bool> visited(N, false);
    queue<int> Queue;
    Queue.push(start_node);
    visited[start_node] = true;

    while(!Queue.empty()){
        int vertex = Queue.front();

        if(print_node)
            cout<<vertex<<" ";

        Queue.pop();

        for(int neighbor : graph[vertex]){
            if(!visited[neighbor]){
                Queue.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }
}

void graph_input(vector<vector<int>> &graph){
    int N, choice = -1;
    cout<<"Enter the size of the graph : ";
    cin>>N;
    graph.resize(N);

    int total_edges;
    cout<<"Enter the no. of Edges : ";
    cin>>total_edges;

    for(int i = 0; i < total_edges; i++){
        int u, v;
        cout<<"Enter the current edge nodes named(0 to n-1): ";
        cin>>u>>v;

        if(u >= N || v >= N){
            cout<<"Nodes beyond the size of graph.\n";
            continue;
        }
        graph[u].push_back(v);
        graph[v].push_back(u);
    }
}

int analysis(std :: function<void()> function){
    auto start = high_resolution_clock::now();
    function();
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    return duration.count();
}

int main(void){
    vector<vector<int>> graph;
    vector<bool> visited;

    double sequential_execution = 0, parellel_execution = 0;
    bool print_node = false;

    int time_taken = 0;
    int num_of_vertices = 1000;
    int num_of_edges = 500000;
    float speed_up = 0.0f;
    bool flag = true;

    while(flag){
        cout<<"1. Sequential BFS \n";
        cout<<"2. Parellel BFS\n";
        cout<<"3. Compare Sequential and parellel BFS with random graph\n";
        cout<<"4. Exit\n";

        int choice = -1;
        cout << "Enter the choice : ";
        cin >> choice;

        switch(choice){
            case 1:
                graph_input(graph);
                print_node = true;

                time_taken = analysis([&] {sequential_BFS(graph, 0, print_node);});

                cout << endl;
                cout << "Time Taken : " << time_taken << endl;
                break;

            case 2:
                graph_input(graph);
                print_node = true;

                time_taken = analysis([&] {parellel_BFS(graph, 0, print_node);});

                cout << endl;
                cout << "Time Taken : " << time_taken << endl;

                break;

            case 3:
                graph.resize(num_of_vertices);

                for(int i = 0; i < num_of_edges; i++){
                    int u = (rand() % num_of_vertices);
                    int v = (rand() % num_of_vertices);

                    graph[u].push_back(v);
                    graph[v].push_back(u);
                }
                print_node = false;
                sequential_execution = analysis([&] {sequential_BFS(graph, 0, print_node);});
                parellel_execution = analysis([&] {parellel_BFS(graph, 0, print_node);});

                speed_up = (float) (sequential_execution/ (float) parellel_execution);

                cout << "Sequential time : "<<sequential_execution<<"ms\n";
                cout << "Parellel time : "<<parellel_execution<<"ms\n";
                cout << "Speed Up : "<<speed_up<<"\n";
                break;
            case 4:
                flag = false;
                break;
            default:
                cout << "Invalid Input \n";
                break;
        }
    }
    return 0;
}

/*
To run this
1. compile using : g++ -fopenmp bfs.cpp
2. run using     : ./a.out
*/