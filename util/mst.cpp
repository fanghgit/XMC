#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <time.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace std;


class Graph{
private:
	int V, E;
	std::vector<std::pair<int, std::pair<int, int> > > edges;
public:
	std::vector<std::pair<int, std::pair<int, int> > > MST;
	Graph(int V, int E);
	void addEdge(int u, int v, int w);
	int kruskalMST();
	void printMST();
};

Graph::Graph(int V, int E)
{
	this->V = V;
	this->E = E;
}

void Graph::addEdge(int u, int v, int w)
{
	edges.push_back( std::make_pair(w, std::make_pair(u,v) ) );
}

void Graph::printMST(){
    std::vector<std::pair<int,std::pair<int,int> > >::iterator it;
		//auto it;
		for(it = MST.begin();it!=MST.end();it++){
        //cout << it->second.first << " - " << it->second.second << endl;
				printf("%d - %d", it->second.first,it->second.second);
    }
}

struct DisjointSet{
    int *parent,*rnk;
    int n;

    DisjointSet(int n){
        this->n = n;
        parent = new int[n+1];
        rnk = new int[n+1];

        for(int i=0;i<=n;i++){
            rnk[i] = 0;
            parent[i] = i;
        }
    }
    int Find(int u){
        if(u != parent[u])
					parent[u] = Find(parent[u]);
        return parent[u];
    }

    void Union(int x,int y){
        // x = Find(x);
        // y = Find(y);
        // if(x != y){
        //     if(rnk[x] < rnk[y]){
        //         rnk[y] += rnk[x];
        //         parent[x] = y;
        //     }
        //     else{
        //         rnk[x] += rnk[y];
        //         parent[y] = x;
        //     }
        // }
				x = Find(x), y = Find(y);

        /* Make tree with smaller height
           a subtree of the other tree  */
        if (rnk[x] > rnk[y])
            parent[y] = x;
        else // If rnk[x] <= rnk[y]
            parent[x] = y;

        if (rnk[x] == rnk[y])
            rnk[y]++;
    }
};

bool comp(const std::pair<int, std::pair<int,int> > &lhs, const std::pair<int, std::pair<int,int> > &rhs)
{
	if(lhs.first < rhs.first)
		return true;
  else if(lhs.first == rhs.first && lhs.second.first == rhs.second.first && lhs.second.second == rhs.second.second)
    return false;
	else if((lhs.first == rhs.first) && (lhs.second.first == 0 || lhs.second.second == 0)) // give priority to label 0
		return true;
	else
		return false;
}

int Graph::kruskalMST(){
    int MSTWeight = 0; //sum of all vertex weights
    printf("before sort, size of edges: %ld\n", edges.size());
    std::sort(edges.begin(),edges.end(), comp);
    printf("sort complete!\n");
		//printf("|E| in kruskal: %d\n", edges.size());
    //for all u in G_v
    //    MAKE-SET(u)
    DisjointSet ds(this->V);

    std::vector<std::pair<int,std::pair<int,int> > >::iterator it;
		//auto it;
		// for all edges in G
    for(it = edges.begin(); it!=edges.end();it++){
        int u = it->second.first;
        int v = it->second.second;

        int setU = ds.Find(u);
        int setV = ds.Find(v);


        if(setU != setV){
            int w = it->first;
            MST.push_back( std::make_pair(w, std::make_pair(u,v) ) );
            MSTWeight += it->first;

            ds.Union(setU,setV);
        }
    }
    return MSTWeight;
}


#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.


int main(int argc, char **argv)
{
  int lower = atoi(argv[1]);
  int upper = atoi(argv[2]);

  FILE* fp;
  fp = fopen("edges.txt", "r");
  int E = 717032598;
  int V = 2812282;
  Graph g(V, E);
  int u;
  int v;  717000000
  int w;
  while(1){
    FSCANF(fp, "%d", &u);
    fscanf(fp, ",");
    FSCANF(fp, "%d", &v);
    fscanf(fp, ",");
    FSCANF(fp, "%d", &w);
    fscanf(fp, "\n");
    g.addEdge( u, v, w );
  }
  printf("adding edges complete\n");

  int weight = g.kruskalMST();
  printf("weight of MST is: %d\n", weight);


  return 0;
}
