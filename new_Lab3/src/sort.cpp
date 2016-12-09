#include <cstdio>
#include <algorithm>

#include <pthread.h>
#include <string.h>
#include <limits.h>
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
	int* array, *array_tmp;
	int left, right, id, middle;
};



void swap(int *a, int *b)
{
	int t = *a;*a = *b;	*b = t;
}






void merge(int* array, int* array_tmp, int left, int mid, int right)
{
	int a = left, b = mid+1;
	for(int i=left;i<=right;++i)
	{
		if((array[a]<array[b] && a <= mid) || b>right)
		{
			array_tmp[i] = array[a++];
		}else
			array_tmp[i] = array[b++];
	}
	memcpy(array + left, array_tmp+left,sizeof(int) * (right-left + 1));
}

void* p_merge(void* args)
{
	Argument* arg = (Argument*) args;
	merge(arg->array, arg->array_tmp, arg->left, arg->middle, arg->right);
}

void* split(void* args)
{
	Argument* arg = (Argument*) args;
	int *array = arg->array;
	int *array_tmp = arg->array_tmp;

	int middle = (arg->left+arg->right)/2;

	Argument arg1, arg2;
	arg1.array = array;
	arg1.array_tmp = array_tmp;
	arg1.left = arg->left;
	arg1.right = middle;
	arg1.id = arg->id;

	arg2.left = middle+1;
	arg2.right = arg->right;
	arg2.array = array;
	arg2.array_tmp = array_tmp;
	arg2.id = arg->id;

	if(arg->left<arg->right)
	{
		split(&arg1);
		split(&arg2);
		merge(array, array_tmp,arg->left,middle,arg->right);
	}
	
	return 0;

}

void mergeSort(int* array, int size)
{
	int d = size / NB_THREADS;

	pthread_t threads[NB_THREADS];
	Argument args[NB_THREADS];
	pthread_mutex_t mutex;

	int *tmp = new int[size];		// temp array
	

	int prev = -1;
	for(int i=0;i<NB_THREADS;i++)
	{
		args[i].left = prev + 1;
		args[i].right = prev + 1 + d;

		if(i == NB_THREADS-1) args[i].right = size-1;

		prev = args[i].right;
		args[i].array = array;
		args[i].id = i;
		args[i].array_tmp = tmp;
		// create the threads
		pthread_create(&threads[i], NULL, &split, (void*)&args[i]);
	}
	
	for(int i=0;i<NB_THREADS;i++)
	{
		pthread_join(threads[i], NULL);
	}

	int idx[NB_THREADS];
	
	

	
	// final merge
	if(NB_THREADS<4)
	{
		// sequential if less than 4 threads
		for(int i=0;i<NB_THREADS;++i) idx[i] = args[i].left;
		for(int i=0;i<size;++i)
		{
			int min_i = -1;
			int min = INT_MAX;
			for(int k=0;k<NB_THREADS;++k)
			{
				if(tmp[idx[k]]<min && idx[k]<=args[k].right)
				{
					min_i = k;
					min = tmp[idx[k]];
				}
			}
			array[i] = min;
			idx[min_i]++;
		}
	}else if(NB_THREADS == 4)
	{
		// if 4 threads : merge with binary tree

		// merge blocks 0 and 1
		args[0].array = array;
		args[0].array_tmp = tmp;
		args[0].left = args[0].left;
		args[0].middle = args[0].right;
		args[0].right = args[1].right;
		pthread_create(&threads[0], NULL, &p_merge, (void*)&args[0]);


		// merge blocks 2 and 3
		args[1].array = array;
		args[1].array_tmp = tmp;
		args[1].left = args[2].left;
		args[1].middle = args[2].right;
		args[1].right = args[3].right;
		pthread_create(&threads[1], NULL, &p_merge, (void*)&args[1]);



		pthread_join(threads[0], NULL);
		pthread_join(threads[1], NULL);

		// final merge of the root
		merge(array, tmp, 0,args[2].left-1, size-1);
		
	}
	
	delete[] tmp;
}


void
sort(int* array, size_t size)
{

	//simple_quicksort(array, size);

//	printf("NB_THREADS = %d\n",NB_THREADS);
#if NB_THREADS == 0
	simple_quicksort(array, size);
#else
	mergeSort(array, size);

#endif // #if NB_THREADS

}
