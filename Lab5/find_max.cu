// Reduction lab, find maximum

#include <stdio.h>

#include "milli.h"



/* first kernel binary search in each block */
__global__ void find_max(int *data, int N)
{
  int i;
  i = threadIdx.x * 2 + blockDim.x*blockIdx.x*2;

  int c = 2;

  while(c <= blockDim.x*2)
  {
	if(i%c == 0)
	{
		data[i] = data[i]>data[i+c/2] ? data[i] : data[i+c/2];
	}
	c*=2;
	__syncthreads();
  }
}

/* second kernel to find the max among the maximum of each block */
__global__ void find_max_among_blocks(int *data,int blockSize, int nbBlocks)
{
	for(int i=0; i<nbBlocks; ++i)
	{
		if(data[0]<data[i*blockSize])
			data[0] = data[i*blockSize];
	}

	
// Write your CUDA kernel here
}

void launch_cuda_kernel(int *data, int N)
{
	// Handle your CUDA kernel launches in this function
	
	int *devdata;
	int size = sizeof(int) * N;
	cudaMalloc( (void**)&devdata, size);
	cudaMemcpy(devdata, data, size, cudaMemcpyHostToDevice );
	
	// Dummy launch
	dim3 dimBlock( 1024, 1 );
	dim3 dimGrid( N / 2048, 1 );
	dim3 dimBlock2(1,1);
	dim3 dimGrid2(1,1);
	find_max<<<dimGrid, dimBlock>>>(devdata, N);

	find_max_among_blocks<<<dimGrid2, dimBlock2>>>(devdata, 2048, N/2048);
	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata, sizeof(int), cudaMemcpyDeviceToHost ); 
	cudaFree(devdata);
}

// CPU max finder (sequential)
void find_max_cpu(int *data, int N)
{
  int i, m;
  
	m = data[0];
	for (i=0;i<N;i++) // Loop over data
	{
		if (data[i] > m)
			m = data[i];
	}
	data[0] = m;
}

//#define SIZE 1024
#define SIZE 2048 * 16384 * 2
// Dummy data in comments below for testing
int data[SIZE];//= {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};

int main()
{


	/*struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	
	printf("max threads pr blocks : %d\n",prop.maxThreadsPerBlock);
	printf("max threads dim : %d %d %d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
	printf("max grid dim : %d %d %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);*/



  // Generate 2 copies of random data
  srand(time(NULL));
  for (long i=0;i<SIZE;i++)
  {
    data[i] = rand() % (SIZE * 5);
    data2[i] = data[i];
  }
  
  // The GPU will not easily beat the CPU here!
  // Reduction needs optimizing or it will be slow.
  ResetMilli();
  find_max_cpu(data, SIZE);
  printf("CPU time %f\n", GetSeconds());
  ResetMilli();
  launch_cuda_kernel(data2, SIZE);
  printf("GPU time %f\n", GetSeconds());

  // Print result
  printf("\n");
  printf("CPU found max %d\n", data[0]);
  printf("GPU found max %d\n", data2[0]);
}
