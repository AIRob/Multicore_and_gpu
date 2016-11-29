// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include "milli.h"

const int N = 1024; 
const int blocksize = 16; 

__global__ 
void simple(float *ca, float *cb, float *cc) 
{
	//int idx = threadIdx.x*blockDim.x + threadIdx.y;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];
	float *ccpu = new float[N*N];

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

	cudaEvent_t myEvent,laterEvent;
	float theTime=0;


	dim3 dimBlock( N, 1 );
	dim3 dimGrid( N, 1 );

	cudaEventCreate(&myEvent);
	cudaEventCreate(&laterEvent);


	cudaEventRecord(myEvent, 0);
	simple<<<dimGrid, dimBlock>>>(ca,cb,cc);
	cudaEventRecord(laterEvent, 0);

	cudaEventSynchronize(laterEvent);
	
	cudaEventElapsedTime(&theTime, myEvent, laterEvent);

	printf("%f\n",theTime);

	cudaThreadSynchronize();
	cudaMemcpy( c, cc, size, cudaMemcpyDeviceToHost ); 
	cudaFree( ca );
	cudaFree( cb );
	cudaFree( cc );
	

	ResetMilli();
	add_matrix(a, b, ccpu, N);
	printf("%lf \n",GetSeconds());// GetMilliseconds(),GetMicroseconds());

	int ok =1;
	for (int i = 0; i < N*N; i++)
	{
		if(c[i] != ccpu[i])
			ok = 0;
	}
	if(ok) printf("OK\n");
	else printf("ERROR\n");
	
	

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] ccpu;
	return EXIT_SUCCESS;
}
