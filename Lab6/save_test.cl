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

	//printf("START KERNEL %d %d : %d %d\n",i,j,ii,jj);
  int k, l;
  unsigned int sumx, sumy, sumz;

	int size = 16+2*KERNELSIZE;
//	printf("----------> size = %d\n",(16+2*KERNELSIZE)*(16+2*KERNELSIZE)*3);
	__local unsigned char mem[(16+2*KERNELSIZE)*(16+2*KERNELSIZE)*3];
	//__local unsigned char mem[16*16*3];
	int init_ii = ii, init_jj = jj;


if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE){
	mem[(ii*size+jj)*3] = image[((i-KERNELSIZE)*n+(j-KERNELSIZE))*3];
	mem[(ii*size+jj)*3+1] = image[((i-KERNELSIZE)*n+(j-KERNELSIZE))*3+1];
	mem[(ii*size+jj)*3+2] = image[((i-KERNELSIZE)*n+(j-KERNELSIZE))*3+2];

	ii+= 2*KERNELSIZE;
	mem[(ii*size+jj)*3] = image[((i+KERNELSIZE)*n+(j-KERNELSIZE))*3];
	mem[(ii*size+jj)*3+1] = image[((i+KERNELSIZE)*n+(j-KERNELSIZE))*3+1];
	mem[(ii*size+jj)*3+2] = image[((i+KERNELSIZE)*n+(j-KERNELSIZE))*3+2];


	ii-=2*KERNELSIZE;
	jj+=2*KERNELSIZE;
	mem[(ii*size+jj)*3] = image[((i-KERNELSIZE)*n+(j+KERNELSIZE))*3];
	mem[(ii*size+jj)*3+1] = image[((i-KERNELSIZE)*n+(j+KERNELSIZE))*3+1];
	mem[(ii*size+jj)*3+2] = image[((i-KERNELSIZE)*n+(j+KERNELSIZE))*3+2];


	ii+=2*KERNELSIZE;
	mem[(ii*size+jj)*3] = image[((i+KERNELSIZE)*n+(j+KERNELSIZE))*3];
	mem[(ii*size+jj)*3+1] = image[((i+KERNELSIZE)*n+(j+KERNELSIZE))*3+1];
	mem[(ii*size+jj)*3+2] = image[((i+KERNELSIZE)*n+(j+KERNELSIZE))*3+2];

	ii-=KERNELSIZE;
	jj-=KERNELSIZE;
}
	/*if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)
	{
	
	if(jj==KERNELSIZE)
	{
//		printf("%d : %d %d\n",ii, 0,KERNELSIZE-1);
		for(k=0;k<KERNELSIZE;++k)
		{
			mem[(ii*size+k)*3+0] = image[(i*n+j-KERNELSIZE+k)*3];
			mem[(ii*size+k)*3+1] = image[(i*n+j-KERNELSIZE+k)*3+1];
			mem[(ii*size+k)*3+2] = image[(i*n+j-KERNELSIZE+k)*3+2];
		}
	}

	if(jj==size-1-KERNELSIZE)
	{	
//		printf("%d : %d %d\n",ii, jj+1, jj+KERNELSIZE);
		for(k=1;k<=KERNELSIZE;++k)
		{
			mem[(ii*size+jj+k)*3] = image[(i*n+j+k)*3];
			mem[(ii*size+jj+k)*3+1] = image[(i*n+j+k)*3+1];
			mem[(ii*size+jj+k)*3+2] = image[(i*n+j+k)*3+2];
		}
	}
	



	if(init_jj==1 && init_ii<KERNELSIZE)
	{
//		printf("%d\n",init_ii);
		for(k=0;k<size;++k)
		{
			mem[(init_ii*size+k)*3] = image[((i-KERNELSIZE)*n+(j-KERNELSIZE+k))*3];
			mem[(init_ii*size+k)*3+1] = image[((i-KERNELSIZE)*n+(j-KERNELSIZE+k))*3+1];
			mem[(init_ii*size+k)*3+2] = image[((i-KERNELSIZE)*n+(j-KERNELSIZE+k))*3+2];
		}
	}


	if(init_jj==1 && init_ii>=16-KERNELSIZE)
	{
// 		printf("----- %d \n",(ii+KERNELSIZE));
		for(k=0;k<size;++k)
		{
			mem[((ii+KERNELSIZE)*size+k)*3] = image[((i+KERNELSIZE)*n+(j-KERNELSIZE+k))*3];
			mem[((ii+KERNELSIZE)*size+k)*3+1] = image[((i+KERNELSIZE)*n+(j-KERNELSIZE+k))*3+1];
			mem[((ii+KERNELSIZE)*size+k)*3+2] = image[((i+KERNELSIZE)*n+(j-KERNELSIZE+k))*3+2];
		}
	}

	
	

	/*if( ii == KERNELSIZE && jj==KERNELSIZE)
	{	
		int a,b;
		int x = i-KERNELSIZE;
		int y = j-KERNELSIZE;
		for(a=0;a<size;a++)
		{
			for(b=0;b<size;b++)
			{
				mem[(a*size+b)*3] = image[((x+a)*n+(y+b))*3];
				mem[(a*size+b)*3+1] = image[((a+x)*n+(y+b))*3+1];
				mem[(a*size+b)*3+2] = image[((x+a)*n+(y+b))*3+2];
			}
		}



	}*/


	


	




	barrier(CLK_LOCAL_MEM_FENCE);
	int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);
	
	if (j < n && i < m) // If inside image
	{
		if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)		//kernel is completely in the image (not partially)
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
