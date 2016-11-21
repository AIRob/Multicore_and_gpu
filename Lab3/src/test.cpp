#include <cstdio>
#include <algorithm>

#include <string.h>

#define NB_THREADS 3
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

struct Argument
{
	int id;
	int *array;
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


#if NB_THREADS > 1

List tab_list[NB_THREADS * NB_THREADS];
List final_list[NB_THREADS];
int *buffer, *buffer2;
int pivots[NB_THREADS-1];
int size_for_thread, size_for_last;

void* thread_sample(void *args)
{
	Argument *arg = (Argument*) args;
	int id = arg->id;
	int *array = arg->array;
	int borne_max = id == NB_THREADS-1 ? size_for_last : size_for_thread;

	//printf("thread %d\n", id);

	for(int i=0;i<borne_max; ++i)
	{
		//printf("%d ", array[i]);
		int j = 0;
		while(j<NB_THREADS-1 && array[i]>pivots[j])
		{
			++j;
		}
		tab_list[NB_THREADS * id + j].push(array[i]);
	}

	//printf("\n");

	/*for(int i=0;i<NB_THREADS;++i)
	{
		simple_quicksort(tab_list[NB_THREADS * id + i].buffer,tab_list[NB_THREADS * id + i].size);
	}*/

	return 0;
}

void* thread_merge(void* args)
{
	Argument *arg = (Argument*) args;
	int id = arg->id;
	int pt = 0;

	for(int i=0;i<NB_THREADS;++i)
	{
		int index = i*NB_THREADS+id;
		if(tab_list[index].size > 0)
		{
			memcpy(final_list[id].buffer + final_list[id].size,tab_list[index].buffer,sizeof(int)*tab_list[index].size);
			final_list[id].size += tab_list[index].size;
		}
		//final_list[id].print();
	}

	//final_list[id].size = pt;
	simple_quicksort(final_list[id].buffer,final_list[id].size);

	return 0;

}
#endif


void
sort(int* array, size_t size)
{
	
	//simple_quicksort(array, size);


#if NB_THREADS == 0
	// Some sequential-specific sorting code
#else
	// Some parallel sorting-related code

	size_for_thread = size / NB_THREADS;
	size_for_last = size - size_for_thread * (NB_THREADS - 1);


	buffer = (int*) malloc(sizeof(int) * NB_THREADS * size );
	for(int i=0;i<NB_THREADS*NB_THREADS; ++i)
	{
		tab_list[i].set_buffer(buffer + i*size_for_thread);
	}

	buffer2 = (int*) malloc(sizeof(int) * size * NB_THREADS);
	for(int i=0;i<NB_THREADS; ++i)
	{
		final_list[i].set_buffer(buffer2 + i*size);
	}



	
	for(int i=0;i<NB_THREADS-1;++i)
	{
		pivots[i] = array[i * size_for_thread];
	}	
	simple_quicksort(pivots,NB_THREADS-1);

	pivots[0] = 7;
	pivots[1] = 17;

	pthread_t threads[NB_THREADS];
	Argument args[NB_THREADS];



    for (int i = 0; i < NB_THREADS; i++) {
	  args[i].id = i;
	  args[i].array = array + size_for_thread * i;
	  pthread_create(&threads[i], NULL, &thread_sample, (void*)&args[i]);
	}


	for (int i = 0; i < NB_THREADS; i++) {
	  pthread_join(threads[i], NULL);
	}

    for (int i = 0; i < NB_THREADS; i++) {
	  pthread_create(&threads[i], NULL, &thread_merge, (void*)&args[i]);
	}


	for (int i = 0; i < NB_THREADS; i++) {
	  pthread_join(threads[i], NULL);
	}



	/*for(int i=0;i<NB_THREADS;i++)
	{
		printf("final list %d\n", i);
		final_list[i].print();
	}*/





	/*for(int i=0;i<NB_THREADS;++i)
	{
		printf("List thread %d :\n",i);
		for(int j=0;j<NB_THREADS;++j)
		{
			tab_list[i*NB_THREADS+j].print();
		}
		printf("\n");
	}*/

	int *pt = array;
	for(int i=0;i<NB_THREADS;++i)
	{
		if(final_list[i].size>0)
		{
			memcpy(pt,final_list[i].buffer,final_list[i].size * sizeof(int));
			pt += final_list[i].size;
		}
	}




	free(buffer);	
	free(buffer2);

#endif // #if NB_THREADS
}


int main()
{
	int array[] = {9, 3, 17, 4, 5, 20, 19, 11, 1, 8, 7, 2, 15, 14, 6};

	sort(array,15);

	printf("\n");
	for(int i=0;i<15;i++)
		printf("%d ", array[i]);

	printf("\nTEST SORT\n");
}