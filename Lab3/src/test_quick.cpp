#include <cstdio>
#include <algorithm>

#include <string.h>

#include <pthread.h>



class List
{
public:
	List() : size(0) {};
	void set_buffer(int *pt) { buffer = pt;}
	void push(int v) { buffer[size++] = v; }

	void print(){
		for(int i=0;i<size;i++)
			printf("%d ",buffer[i]);
		printf("\n");
	}


	int* buffer;
	int size;
};





// A very simple quicksort implementation
// * Recursion until array size is 1
// * Bad pivot picking
// * Not in place
static
void
simple_quicksort(int *array, size_t size)
{
	int pivot, pivot_count, i;
	int *left, *right;
	size_t left_size = 0, right_size = 0;

	pivot_count = 0;

	// This is a bad threshold. Better have a higher value
	// And use a non-recursive sort, such as insert sort
	// then tune the threshold value
	if(size > 1)
	{
		// Bad, bad way to pick a pivot
		// Better take a sample and pick
		// it median value.
		pivot = array[size / 2];
		
		left = (int*)malloc(size * sizeof(int));
		right = (int*)malloc(size * sizeof(int));

		// Split
		for(i = 0; i < size; i++)
		{
			if(array[i] < pivot)
			{
				left[left_size] = array[i];
				left_size++;
			}
			else if(array[i] > pivot)
			{
				right[right_size] = array[i];
				right_size++;
			}
			else
			{
				pivot_count++;
			}
		}

		// Recurse		
		simple_quicksort(left, left_size);
		simple_quicksort(right, right_size);

		// Merge
		memcpy(array, left, left_size * sizeof(int));
		for(i = left_size; i < left_size + pivot_count; i++)
		{
			array[i] = pivot;
		}
		memcpy(array + left_size + pivot_count, right, right_size * sizeof(int));

		// Free
		free(left);
		free(right);
	}
	else
	{
		// Do nothing
	}
}






struct Argument
{
	int* array;
	int left, right, id;
};



void swap(int *a, int *b)
{
	int t = *a;*a = *b;	*b = t;
}


#define NB_THREADS 5

pthread_t threads[NB_THREADS];
Argument arguments[NB_THREADS];
pthread_mutex_t mutex;

int threads_used = 0; //NB_THREADS;

void* p_quicksort(void *args)
{
	Argument*  arg = (Argument *) args;

	int left = arg->left;
	int init_left = left;
	int right = arg->right;
	int init_right = right;
	int* array = arg->array;
	int id = arg->id;
	if(left>=right) return 0;



	int pivot_i = (left+right)/2;
	int pivot = array[pivot_i];



	int l = left;
	int r = right;
	int *t = array;

	int m=(l+r)/2;


	//printf("\n---- Thread : %d -----\n",id);
	//printf("pivot %d\n",array[m]);
	//for(int i=left;i<=right;i++)
	//{
	//	printf("%d ", array[i]);
	//}
	//printf("\n");


	swap(array + init_left, array + pivot_i);

	++left;
	while(left<=right)
	{
		
		while(left<init_right && array[left]<array[init_left])	left++;
		while(array[right]>array[init_left])	right--;
		if(right>left)	swap(array + (left++),array + (right--));
		else left++;
		
	}
	swap(array + init_left,array + right);



	for(int i=init_left;i<=init_right;i++)
	{
		printf("%d ", array[i]);
	}
	printf("\n");

	printf("p[right] = %d\n",array[right]);


	if(right-1-init_left>=1)
	{
		if(threads_used<NB_THREADS)
		{

			pthread_mutex_lock(&mutex);
			arguments[threads_used].left = init_left;
			arguments[threads_used].right = right-1;

		  	pthread_create(&threads[threads_used], NULL, &p_quicksort, (void*)&arguments[threads_used]);
			++threads_used;
			pthread_mutex_unlock(&mutex);


		}else
		{

			arguments[id].left = init_left;
			arguments[id].right = right-1;

			//printf("Recursive entre %d et %d\n",t[init_left],t[right-1]);

			printf("no more threads : recursive : ");
			for(int i=init_left;i<=right-1;++i) printf("%d ",array[i] );
			printf("\n");

			p_quicksort(arguments + id);
		}
	}

	/*if(threads_used<NB_THREADS)
	{
		//pthread_mutex_lock(&mutex);

		arguments[remaining_threads-1].left = right+1;
		arguments[remaining_threads-1].right = init_right;

	  	pthread_create(&threads[remaining_threads-1], NULL, &p_quicksort, (void*)&arguments[remaining_threads-1]);
	 	remaining_threads--;
		//pthread_mutex_unlock(&mutex);

	}else*/
	if(init_right-(right+1)>=1)
	{

		arguments[id].left = right+1;
		arguments[id].right = init_right;
		//printf("no more threads : recursive : ");

		printf("Recursive entre %d et %d\n",t[right+1],t[init_right]);
		for(int i=right+1;i<=init_right;++i) printf("%d ",array[i] );
		printf("\n");


		p_quicksort(arguments + id);
	}
}








void
sort(int* array, size_t size)
{
	
	//simple_quicksort(array, size);

	if(pthread_mutex_init(&mutex,NULL)!=0)
		printf("\n\nERROROR MUTEX\n\n");
#if NB_THREADS == 0
	// Some sequential-specific sorting code
#else
	// Some parallel sorting-related code

	for(int i=0;i<NB_THREADS;i++)
	{
		arguments[i].array = array;
		arguments[i].id = i;
	}

	arguments[0].left = 0;
	arguments[0].right = size-1;

	++threads_used;

 	pthread_create(&threads[0], NULL, &p_quicksort, (void*)&arguments[0]);

	for(int i=0;i<NB_THREADS;++i)
	{
		pthread_join(threads[i],NULL);
	}
#endif // #if NB_THREADS

	pthread_mutex_destroy(&mutex);
}
	

int main()
{
	int array[] = {9, 3, 17, 4, 5, 20, 19, 11, 1, 8, 7, 2, 15, 14, 6, 32 ,26 ,7 ,45 ,16 ,10 ,19 ,12 ,2 ,33};

	sort(array,25);

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