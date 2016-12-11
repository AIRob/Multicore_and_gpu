/*
 * Image filter in OpenCL
 */

#define KERNELSIZE 5

__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int n, const unsigned int m)
{ 
	unsigned int j = get_global_id(1) % 512;
	unsigned int i = get_global_id(0) % 512;

	unsigned ii = get_local_id(0);
	unsigned jj = get_local_id(1);


  	int k, l;
	unsigned int sumx, sumy, sumz;


	int size = 16+2*KERNELSIZE;	// width and height of the local memory


	__local unsigned char mem[(16+2*KERNELSIZE)*(16+2*KERNELSIZE)*3];


	if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)
	{

		mem[(ii*size+jj)*3] = image[((i-KERNELSIZE)*n+(j-KERNELSIZE))*3];
		mem[(ii*size+jj)*3+1] = image[((i-KERNELSIZE)*n+(j-KERNELSIZE))*3+1];
		mem[(ii*size+jj)*3+2] = image[((i-KERNELSIZE)*n+(j-KERNELSIZE))*3+2];

		jj+=2*KERNELSIZE;
		mem[(ii*size+jj)*3] = image[((i-KERNELSIZE)*n+(j+KERNELSIZE))*3];
		mem[(ii*size+jj)*3+1] = image[((i-KERNELSIZE)*n+(j+KERNELSIZE))*3+1];
		mem[(ii*size+jj)*3+2] = image[((i-KERNELSIZE)*n+(j+KERNELSIZE))*3+2];

		jj-=2*KERNELSIZE;
		ii+= 2*KERNELSIZE;
		mem[(ii*size+jj)*3] = image[((i+KERNELSIZE)*n+(j-KERNELSIZE))*3];
		mem[(ii*size+jj)*3+1] = image[((i+KERNELSIZE)*n+(j-KERNELSIZE))*3+1];
		mem[(ii*size+jj)*3+2] = image[((i+KERNELSIZE)*n+(j-KERNELSIZE))*3+2];


		jj+=2*KERNELSIZE;
		mem[(ii*size+jj)*3] = image[((i+KERNELSIZE)*n+(j+KERNELSIZE))*3];
		mem[(ii*size+jj)*3+1] = image[((i+KERNELSIZE)*n+(j+KERNELSIZE))*3+1];
		mem[(ii*size+jj)*3+2] = image[((i+KERNELSIZE)*n+(j+KERNELSIZE))*3+2];

		ii-=KERNELSIZE;
		jj-=KERNELSIZE;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);
	
	if (j < n && i < m) // If inside image
	{
		if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)		//the pixel is not on a border
		{

			// Filter kernel
			sumx=0;sumy=0;sumz=0;					
			for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
				for(l=-KERNELSIZE;l<=KERNELSIZE;l++)	
				{
					//printf("%d\n",((ii+k)*16+(jj+l))*3+0);
						sumx += mem[((ii+k)*size+(jj+l))*3+0];
						sumy += mem[((ii+k)*size+(jj+l))*3+1];
						sumz += mem[((ii+k)*size+(jj+l))*3+2];
				}
			out[(i*n+j)*3+0] = sumx/divby;
			out[(i*n+j)*3+1] = sumy/divby;
			out[(i*n+j)*3+2] = sumz/divby;
		}
		else
		// Edge pixels are not filtered
		{
			out[(i*n+j)*3+0] = image[(i*n+j)*3+0];
			out[(i*n+j)*3+1] = image[(i*n+j)*3+1];
			out[(i*n+j)*3+2] = image[(i*n+j)*3+2];
		}
	}
}
