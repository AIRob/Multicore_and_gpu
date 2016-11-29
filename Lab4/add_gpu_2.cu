// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
void simple(float *ca, float *cb, float *cc) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//int idx = threadIdx.x*N + threadIdx.y;
	cc[idx] = ca[idx] + cb[idx];
}


void add_matrix(float *a, float *b, float *c, int N)
{
	int index;
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

int main()
{
	float a[N*N];
	float b[N*N];
	float c[N*N];
	float ccpu[N*N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}
	float *ca,*cb,*cc;

	const int size = N*N*sizeof(float);
	cudaMalloc( (void**)&ca, size );
	cudaMemcpy( ca, a, size, cudaMemcpyHostToDevice ); 
	cudaMalloc( (void**)&cb, size );
	cudaMemcpy( cb, b, size, cudaMemcpyHostToDevice ); 
	cudaMalloc( (void**)&cc, size );

	
	

	dim3 dimBlock( N,1 );
	dim3 dimGrid( N,1 );
	simple<<<dimGrid, dimBlock>>>(ca,cb,cc);
	cudaThreadSynchronize();
	cudaMemcpy( c, cc, size, cudaMemcpyDeviceToHost ); 
	cudaFree( ca );
	cudaFree( cb );
	cudaFree( cc );
	


	add_matrix(a, b, ccpu, N);

	int ok =1;
	for (int i = 0; i < N*N; i++)
	{
		if(c[i] != ccpu[i])
			ok = 0;
	}
	if(ok) printf("OK\n");
	else printf("ERROR\n");
	

	return EXIT_SUCCESS;
}
