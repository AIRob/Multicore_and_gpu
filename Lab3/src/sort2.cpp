#include <cstdio>
#include <algorithm>

#include <pthread.h>
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
	int* array;
	int left, right, id;
};



void swap(int *a, int *b)
{
	int t = *a;*a = *b;	*b = t;
}




pthread_t threads[NB_THREADS];
Argument arguments[NB_THREADS];
pthread_mutex_t mutex;

int threads_used = 0;

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



	//pivot choice : median of first middle and last values
	int a = array[left];
	int b = array[(left+right)/2];
	int c = array[right];

	int pivot_i;

	if(a>=b && a>=c)
		pivot_i	= b>=c? (left+right)/2 : right;
	else if(b>=a && b>=c)
		pivot_i = a>=c? left : right;
	else
		pivot_i = a>=b? left : (left+right)/2;

	int pivot = array[pivot_i];


	// place the pivot at the beginning
	swap(array + init_left, array + pivot_i);

	++left;
	while(left<=right)
	{
		
		while(left<init_right && array[left]<array[init_left])	left++;
		while(array[right]>array[init_left])	right--;
		if(right>left)	swap(array + (left++),array + (right--));
		else left++;
		
	}
	// replace the pivot at the correct place
	swap(array + init_left,array + right);



	if(right-1-init_left>=1)
	{
		// if there is a thread free
		if(threads_used<NB_THREADS)
		{	//give this part of the array to another thread
			pthread_mutex_lock(&mutex);			//to protect the variable threads_used
			arguments[threads_used].left = init_left;
			arguments[threads_used].right = right-1;
		  	pthread_create(&threads[threads_used], NULL, &p_quicksort, (void*)&arguments[threads_used]);
			++threads_used;
			pthread_mutex_unlock(&mutex);
		}else	
		{	//treat this part of the array recursively

			arguments[id].left = init_left;
			arguments[id].right = right-1;
			p_quicksort(arguments + id);
		}
	}

	// right part always recursively in the same thread
	if(init_right-(right+1)>=1)
	{

		arguments[id].left = right+1;
		arguments[id].right = init_right;
		p_quicksort(arguments + id);
	}
}




void
sort(int* array, size_t size)
{
	
	//simple_quicksort(array, size);

#if NB_THREADS == 0
	simple_quicksort(array, size);	
#else

	if(pthread_mutex_init(&mutex,NULL)!=0)
		printf("\n\nERROROR MUTEX\n\n");


	//init arguments
	for(int i=0;i<NB_THREADS;i++)
	{
		arguments[i].array = array;
		arguments[i].id = i;
	}

	// launch the first thread
	arguments[0].left = 0;
	arguments[0].right = size-1;
	++threads_used;
 	pthread_create(&threads[0], NULL, &p_quicksort, (void*)&arguments[0]);


 	// wait all threads
	for(int i=0;i<NB_THREADS;++i)
	{
		pthread_join(threads[i],NULL);
	}
	pthread_mutex_destroy(&mutex);

#endif // #if NB_THREADS

}
