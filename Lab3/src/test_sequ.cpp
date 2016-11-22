#include <cstdio>
#include <algorithm>

#include <string.h>




void merge(int* array,int* array_tmp, int left, int mid, int right)
{

	int a = left, b = mid+1;
	for(int i=left;i<=right;++i)
	{
		if((array[a]<array[b] && a <= mid) || b>right)
		{
			array_tmp[i] = array[a++];
		}else
			array_tmp[i] = array[b++];
		//array_tmp[i] = array[a]<array[b] ? array[a++] : array[b++];
	}
	memcpy(array + left, array_tmp+left,sizeof(int) * (right-left + 1));
}


void sort(int* array, int* array_tmp, int left, int right)
{
	if(left<right)
	{
		int middle = (left+right)/2;
		sort(array, array_tmp,left,middle);
		sort(array, array_tmp,middle+1,right);
		merge(array, array_tmp,left,middle,right);
	}
}
	

int main()
{
	int array[] = {9, 3, 17, 4, 5, 20, 19, 11, 1, 8, 7, 2, 15, 14, 6, 32 ,26 ,7 ,45 ,16 ,10 ,19 ,12 ,2 ,33};
	int array2[25]; 

	sort(array, array2,0,24);

	printf("\n");

	for(int i=0;i<25;i++)
		printf("%d ", array[i]);


	printf("\nTEST SORT\n");

	for(int i=1;i<25;i++)
		if(array[i-1]>array[i])
		{
			printf("===== ERROR ======\n");
			return 0;
		}

	printf("====== OK ======\n");

}