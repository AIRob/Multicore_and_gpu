// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <unistd.h>
#include "milli.h"

const int N = 1024; 		//1024	//50
const int DIV = 64;		//32	//10
//const int blocksize = 16; 

void clear_data(float *data, int size)
{
	for(int i=0;i<size;i++)
		data[i] = 0;	
}

__global__ 
void simple(float *ca, float *cb, float *cc) 
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;


	int idx = N * X + Y;
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

	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	
	printf("max threads pr blocks : %d\n",prop.maxThreadsPerBlock);
	printf("max threads dim : %d %d %d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
	printf("max grid dim : %d %d %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);


	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];
	float *ccpu = new float[N*N];

	clear_data(c,N*N);
	clear_data(ccpu,N*N);

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
	cudaMemcpy( cc, c, size, cudaMemcpyHostToDevice );

	cudaEvent_t myEvent,laterEvent;
	float theTime=0;

	printf("blockDim : %d x %d\n",N/DIV, N/DIV);
	printf("gridDim : %d x %d\n",DIV,DIV);
	dim3 dimBlock(N/DIV, N/DIV );
	dim3 dimGrid(DIV, DIV );

	cudaEventCreate(&myEvent);
	cudaEventCreate(&laterEvent);


	cudaEventRecord(myEvent, 0);
//	usleep(1000);
	simple<<<dimGrid, dimBlock>>>(ca,cb,cc);
	cudaEventRecord(laterEvent, 0);

	cudaEventSynchronize(laterEvent);
	
	cudaEventElapsedTime(&theTime, myEvent, laterEvent);

	printf("%lf\n",(double)theTime/1000);

	cudaThreadSynchronize();
	cudaMemcpy( c, cc, size, cudaMemcpyDeviceToHost ); 
	cudaFree( ca );
	cudaFree( cb );
	cudaFree( cc );
	

	ResetMilli();

	add_matrix(a, b, ccpu, N);
//	printf("%d \n", GetMilliseconds()); //GetMicroseconds());
	printf("%lf \n",GetSeconds());

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
