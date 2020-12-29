#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

inline void read(int &x)
{
	x=0;char c=getchar();
	while(c<'0' || c>'9')c=getchar();
	while(c>='0' && c<='9')
        {
		x=x*10+c-'0';
		c=getchar();
	} 
}


void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
	int world_rank, world_size, as, bs;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if(world_rank == 0){
		scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
	}
	MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

	as = (*n_ptr)*(*m_ptr);
	bs = (*m_ptr)*(*l_ptr);
	*a_mat_ptr = (int*)calloc(as, sizeof(int));
	*b_mat_ptr = (int*)calloc(bs, sizeof(int));

	if(world_rank == 0){
		int i, j, idx;
		for(i = 0 ; i < as ; i++){
			//scanf("%d", *a_mat_ptr+i);
			read(*(*a_mat_ptr+i));
		}
		for(i = 0 ; i < *m_ptr ; i++){
			for(j = 0 ; j < *l_ptr ; j++){
				//scanf("%d", *b_mat_ptr+i+j*(*m_ptr));
				read(*(*b_mat_ptr+i+j*(*m_ptr)));
			}
		}
	}

	MPI_Bcast(*a_mat_ptr, as, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(*b_mat_ptr, bs, MPI_INT, 0, MPI_COMM_WORLD);
}


void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int *C, *ans;
	int len, start, end;
	len = n*l;
	start = (n/world_size)*world_rank;
	end = start+(n/world_size);
	if (world_rank == world_size-1){
		end = n;
	}

	C = (int*)calloc(len, sizeof(int));
	ans = (int*)calloc(len, sizeof(int));
	
	int sum, i, j, k, a_idx, b_idx, c_idx;//, tmp, tc;
	//c_idx = start*l;
	a_idx = start*m;
	
	for(i = start ; i < end ; i++, a_idx += m){
		c_idx= i*l;
		//a_idx = i*m;
		//tc = c_idx;
		b_idx = 0;
		for(j = 0 ; j < l ; j++){
			sum = 0;
			for(k = 0 ; k < m ; k++){
				sum += a_mat[a_idx+k]*b_mat[b_idx];
				//sum += tmp;
				b_idx++;
			}
			C[c_idx] = sum;
			c_idx++;
			//C[tc] = sum;
			//tc++;
		}
	}
	
	MPI_Reduce(C, ans, len, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(world_rank == 0){
		for(i = 0 ; i < n ; i++){
			c_idx = i*l;
			for(j = 0 ; j < l ; j++){
				if(j) putchar(' ');
				printf("%d", ans[c_idx]);
				c_idx++;
			}
			//printf("\n");
			putchar('\n');
		}
	}
}

void destruct_matrices(int *a_mat, int *b_mat){
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	if(world_rank == 0){
		free(a_mat);
		free(b_mat);
	}
}
