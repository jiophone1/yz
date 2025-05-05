#include <bits/stdc++.h> 
#include <ctime> 
#include <omp.h>
#include<atomic>
using namespace std;


using std :: chrono :: duration_cast;
using std :: chrono :: high_resolution_clock; 
using std :: chrono :: milliseconds;

void DFS(vector<vector<int>> &graph, vector<bool> &visited_bool, int node, bool &print_node) {
    int n = graph.size();
    // 1) Build an atomic visited array
    vector<atomic<bool>> visited_atomic(n);
    for (int i = 0; i < n; ++i)
        visited_atomic[i] = visited_bool[i];

    // 2) Mark and print the root
    if (!visited_atomic[node].exchange(true)) {
        if (print_node) {
            #pragma omp critical
            cout << node << " ";
        }
    }

    // 3) Launch one task per neighbor of 'node'
    #pragma omp parallel
    #pragma omp single
    {
        for (int nei : graph[node]) {
            // try to claim this neighbor
            if (!visited_atomic[nei].exchange(true)) {
                #pragma omp task firstprivate(nei)
                {
                    // 4) Do a pure sequential DFS from 'nei'
                    stack<int> st;
                    st.push(nei);
                    while (!st.empty()) {
                        int v = st.top();
                        st.pop();

                        if (print_node) {
                            #pragma omp critical
                            cout << v << " ";
                        }

                        for (int w : graph[v]) {
                            // only push unseen nodes
                            if (!visited_atomic[w].exchange(true)) {
                                st.push(w);
                            }
                        }
                    }
                }
            }
        }
        #pragma omp taskwait
    }

    // 5) Copy back into the original bool array
    for (int i = 0; i < n; ++i)
        visited_bool[i] = visited_atomic[i];
}


void DFS_with_threads(vector<vector<int>> &graph, int start, bool &print_node){
    int N = graph.size();
    vector<bool> visited(N, false);
    DFS(graph, visited, start, print_node);
}

void DFS_without_threads(vector<vector<int>> &graph, vector<bool> & visited, int node, bool &print_node) {
    visited[node] = true;

    if(print_node)
        cout << node << " ";

    for(int i = 0; i < graph[node].size(); i++){
        if(!visited[graph[node][i]])
            DFS_without_threads(graph, visited, graph[node][i], print_node);
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
    }
}

int analysis(std :: function<void()> function){
    auto start = high_resolution_clock::now();
    function();
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    return duration.count();
}

int main(){
    vector<vector<int>> graph;
    vector<bool> visited;
    int N = 0;

    double execution1 = 0, execution2 = 0;
    bool print_node = false;

    int time_taken = 0;
    int num_of_vertices = 1000;
    int num_of_edges = 500000;
    float speed_up = 0.0f;

    bool flag = true;

    while(flag){
        cout<<"1. Sequential DFS \n";
        cout<<"2. Parallel DFS\n";
        cout<<"3. Compare Sequential and parellel DFS with random graph\n";
        cout<<"4. Exit\n";

        int choice = -1;
        cout << "Enter the choice : ";
        cin >> choice;

        switch(choice){
            case 1:
                graph_input(graph);
                print_node = true;

                N = graph.size();
                visited.resize(N, false);

                time_taken = analysis([&] {DFS_without_threads(graph, visited, 0, print_node);});
                cout << endl;
                cout << "Time taken : "<<time_taken << "\n";
            
                break;
            
            case 2:
                graph_input(graph);
                print_node = true;

                time_taken = analysis([&] {DFS_with_threads(graph, 0, print_node);});

                cout<<endl;
                cout<<"Time taken : "<<time_taken<<endl;

                break;

            case 3:
                graph.resize(num_of_vertices);
                for(int i = 0; i < num_of_edges; i++){
                    int u = (rand()% num_of_vertices);
                    int v = (rand()% num_of_vertices);

                    graph[u].push_back(v);
                    graph[v].push_back(u);
                }

                N = graph.size();
                visited.resize(N, false);

                print_node = false;
                execution1 = analysis([&] {DFS_without_threads(graph, visited, 0, print_node);});
                execution2 = analysis([&] {DFS_with_threads(graph, 0, print_node);});

                cout << "Sequential time : "<< execution1 << "ms" << endl;
                cout << "Parellel time : "<< execution2 <<"ms"<<endl;
                cout << "Speed Up : "<<speed_up<<endl;

                graph.clear();
                break;

            case 4:
                flag = false;
                break;

            default:

                cout<<"Invalid Input " << endl;
                break;
        }
    }
    return 0;
}


/*
To run this
1. compile using : g++ -fopenmp dfs.cpp
2. run using     : ./a.out
*/