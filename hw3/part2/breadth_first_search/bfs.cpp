#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances, int num_nodes, int num_edges, 
    int *outgoing_starts, int *outgoing_edges, int *change, bool *bu_frontier)
{
	#pragma omp parallel for schedule(dynamic, 1024)
	for (int i = 0; i < frontier->count; i++)
    	{
    		int node = frontier->vertices[i];
        	int start_edge = outgoing_starts[node];
        	int end_edge = (node == num_nodes - 1)? num_edges : outgoing_starts[node + 1];

		// attempt to add all neighbors to the new frontier
        	for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        	{
        		int outgoing = outgoing_edges[neighbor];
            		if (distances[outgoing] == NOT_VISITED_MARKER){
		    		if(__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node]+1))
				{
                			//int index = new_frontier->count++;
                			int index = __sync_add_and_fetch(&new_frontier->count, 1);
					new_frontier->vertices[index-1] = outgoing;
					bu_frontier[outgoing] = 1;
					*change = 1;
				}
            		}
        	}
    	}
}

bool BU_step(bool *frontier, bool *new_frontier, int *distances, int d, int total_nodes, int total_edges, int *incoming_starts, int *incoming_edges)
{
	bool change = false;

	#pragma omp parallel for schedule(dynamic, 1024)
	for (int i = 0 ; i < total_nodes ; i++)
    	{
    		if(distances[i] == NOT_VISITED_MARKER)
		{
			int start_edge = incoming_starts[i];
	        	int end_edge = (i == total_nodes - 1)? total_edges : incoming_starts[i + 1];
			
			for (int neighbor = start_edge ; neighbor < end_edge ; neighbor++)
	        	{
	            		int incoming = incoming_edges[neighbor];
				if (frontier[incoming])
				{
					new_frontier[i] = 1;
	                		distances[i] = d;
					change = 1;
					break;
	            		}
	        	}
        	}
	}
	return change;
}

void one_BU_step(bool *frontier, bool *new_frontier, int *distances, int d, int total_nodes, int total_edges, int *incoming_starts, int *incoming_edges, int *change)
{
        #pragma omp parallel for schedule(dynamic, 1024)
        for (int i = 0 ; i < total_nodes ; i++)
        {
                if(distances[i] == NOT_VISITED_MARKER)
                {
                        int start_edge = incoming_starts[i];
                        int end_edge = (i == total_nodes - 1)? total_edges : incoming_starts[i + 1];

                        for (int neighbor = start_edge ; neighbor < end_edge ; neighbor++)
                        {
                                int incoming = incoming_edges[neighbor];
                                if (frontier[incoming])
                                {
                                        new_frontier[i] = 1;
                                        distances[i] = d;
                                        *change = 1;
                                        break;
                                }
                        }
                }
        }
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    bool *bu_frontier = (bool*)calloc(graph->num_nodes, sizeof(bool));
    int change = 0;

    // initialize all nodes to NOT_VISITED
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(frontier, new_frontier, sol->distances, graph->num_nodes, graph->num_edges, graph->outgoing_starts, graph->outgoing_edges, &change, bu_frontier);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}


void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    
	bool *frontier = (bool*)calloc(graph->num_nodes, sizeof(bool));
	bool *new_frontier = (bool*)calloc(graph->num_nodes, sizeof(bool));

	#pragma omp parallel for
	for (int i = 0 ; i < graph->num_nodes ; i++)
    	sol->distances[i] = NOT_VISITED_MARKER;

    sol->distances[ROOT_NODE_ID] = 0;
	frontier[ROOT_NODE_ID] = 1;
	
	int dist = 1, nn = graph->num_nodes, ne = graph->num_edges;
	int *ins = graph->incoming_starts;
	int *ine = graph->incoming_edges;
	bool change = 1;

   	while(change){
		change = BU_step(frontier, new_frontier, sol->distances, dist, nn, ne, ins, ine);
		
		bool *tmp = frontier;
		frontier = new_frontier;
		new_frontier = tmp;
		dist++;
	}
}


void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    	vertex_set list1;
    	vertex_set list2;
    	vertex_set_init(&list1, graph->num_nodes);
    	vertex_set_init(&list2, graph->num_nodes);

    	vertex_set *frontier = &list1;
    	vertex_set *new_frontier = &list2;

    	// initialize all nodes to NOT_VISITED
    	#pragma omp parallel for
    	for (int i = 0; i < graph->num_nodes; i++)
        	sol->distances[i] = NOT_VISITED_MARKER;

    	// setup frontier with the root node
    	frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    	sol->distances[ROOT_NODE_ID] = 0;

	bool *bu_frontier = (bool*)calloc(graph->num_nodes, sizeof(bool));
	bool *bu_new_frontier = (bool*)calloc(graph->num_nodes, sizeof(bool));

    	int dist = 1, nn = graph->num_nodes, ne = graph->num_edges;
        int *ins = graph->incoming_starts;
        int *ous = graph->outgoing_starts;
	int *ine = graph->incoming_edges;
	int *oue = graph->outgoing_edges;
	int change = 1;

        bool using_bot = 0;

	while(change){
		change = 0;
		if(!using_bot && (float)(frontier->count)/(float)(nn) < 0.1){
			top_down_step(frontier, new_frontier, sol->distances, nn, ne, ous, oue, &change, bu_frontier);
		}
		else{
			one_BU_step(bu_frontier, bu_new_frontier, sol->distances, dist, nn, ne, ins, ine, &change);
			using_bot = 1;
		}
		vertex_set *tmp = frontier;
		frontier = new_frontier;
		new_frontier = tmp;

		if(using_bot){
			bool *tmp = bu_frontier;
			bu_frontier = bu_new_frontier;
			bu_new_frontier = tmp;

		}
		dist++;
	}
}
