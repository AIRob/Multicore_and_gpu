#include <cstdio>
#include <algorithm>

#include <string.h>

#include "sort.h"

// These can be handy to debug your code through printf. Compile with CONFIG=DEBUG flags and spread debug(var)
// through your code to display values that may understand better why your code may not work. There are variants
// for strings (debug()), memory addresses (debug_addr()), integers (debug_int()) and buffer size (debug_size_t()).
// When you are done debugging, just clean your workspace (make clean) and compareile with CONFIG=RELEASE flags. When
// you demonstrate your lab, please cleanup all debug() statements you may use to faciliate the reading of your code.
#if defined DEBUG && DEBUG != 0
int *begin;
#define debug(var) printf("[%s:%s:%d] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#else
#define show(first, last)
#define show_ptr(first, last)
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif









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





// A C++ container class that translate int pointer
// into iterators with little constant penalty
template<typename T>
class DynArray
{
	typedef T& reference;
	typedef const T& const_reference;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	public:
	DynArray(T* buffer, size_t size)
	{
		this->buffer = buffer;
		this->size = size;
	}

	iterator begin()
	{
		return buffer;
	}

	iterator end()
	{
		return buffer + size;
	}

	protected:
		T* buffer;
		size_t size;
};

static
void
cxx_sort(int *array, size_t size)
{
	DynArray<int> cppArray(array, size);
	std::sort(cppArray.begin(), cppArray.end());
}

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



List tab_list[NB_THREADS * NB_THREADS];
int *buffer;
int pivots[NB_THREADS-1];
int size_for_thread, size_for_last;

void* thread_sample(void *args)
{
	Argument *arg = (Argument*) args;
	int id = arg->id;
	int *array = arg->array;
	int borne_max = id == NB_THREADS-1 ? size_for_last : size_for_thread;
	for(int i=0;i<borne_max; ++i)
	{
		int j = 0;
		while(j<NB_THREADS-1 && array[i]>=pivots[j])
		{
			++j;
		}
		tab_list[NB_THREADS * id + j].push(array[i]);
	}

	return 0;
}



void
sort(int* array, size_t size)
{
	
	simple_quicksort(array, size);


#if NB_THREADS == 0
	// Some sequential-specific sorting code
#else
	// Some parallel sorting-related code

	size_for_thread = size / NB_THREADS;
	size_for_last = size - size_for_thread * (NB_THREADS - 1);


	buffer = (int*) malloc(sizeof(int) * NB_THREADS * size);
	for(int i=0;i<NB_THREADS*NB_THREADS; ++i)
	{
		tab_list[i].set_buffer(buffer + i*size_for_thread);
	}

	
	//printf("Pivots : ");
	for(int i=0;i<NB_THREADS-1;++i)
	{
		pivots[i] = array[i * size_for_thread];
		//printf("%d ",pivots[i]);
	}	
	//printf("\n");


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
	for(int i=0;i<NB_THREADS*NB_THREADS;++i)
	{
		if(tab_list[i].size>0)
		{
			memcpy(pt,tab_list[i].buffer,tab_list[i].size * sizeof(int));
			pt += tab_list[i].size;
		}
	
	}


	free(buffer);	

#endif // #if NB_THREADS
}
