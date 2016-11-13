/*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5



stack_t *stack;
data_t data;

#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;

    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
      {
        // See how fast your implementation can pop MAX_PUSH_POP elements in parallel
        stack_pop(stack);
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

    return NULL;
  }
#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
    {
        // See how fast your implementation can push MAX_PUSH_POP elements in parallel
      stack_push(stack,i+args->id);
    }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#endif
#endif

/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
test_init()
{
  // Initialize your test batch
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;

  // Allocate a new stack and reset its values
  stack = malloc(sizeof(stack_t));

  // Reset explicitely all members to a well-known initial value
  // For instance (to be deleted as your stack design progresses):
  stack->head = 0;
}

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks
  free(stack);
}

void
test_finalize()
{
  // Destroy properly your test batch
}


// function called by each thread to test concurrent push
void* thread_push(void * arg)
{
  int i=0;
  for(i=0; i<MAX_PUSH_POP / NB_THREADS; ++i)
  {
    stack_push(stack, i);
  }
  return 0;
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it

  int i;
  pthread_t thread[NB_THREADS];

  for (i = 0; i < NB_THREADS; i++) {
    pthread_create(&thread[i], NULL, &thread_push, NULL);
  }
  
  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(thread[i], NULL);
  }

  // Do some work
  //stack_push(stack,1/* add relevant arguments here */);

  // check if the stack is in a consistent state
  stack_check(stack);

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  assert(stack->head != 0);

  int sum = 0;
  while(stack->head)
  {
    sum += stack_pop(stack);
  }

  int temp = (MAX_PUSH_POP/NB_THREADS);
  int result = NB_THREADS * ((temp-1)*temp / 2);

  //printf("sum = %d\n", sum);
  // For now, this test always fails
  return (sum == result);
}

void* thread_pop(void* arg)
{
  int i;
  for(i=0;i<MAX_PUSH_POP/NB_THREADS;i++)
  {
    stack_pop(stack);
  }
  return 0;
}

int
test_pop_safe()
{
  // Same as the test above for parallel pop operation
  int i=0;
  for(i=0;i<(MAX_PUSH_POP/NB_THREADS)*NB_THREADS;i++)
  {
    stack_push(stack,i);
  }
  //printf("size = %d\n", stack_size(stack));

  pthread_t thread[NB_THREADS];

  for (i = 0; i < NB_THREADS; i++) {
    pthread_create(&thread[i], NULL, &thread_pop, NULL);
  }
  
  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(thread[i], NULL);
  }

  //printf("size = %d\n", stack_size(stack));

  return stack->head == NULL;
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

#if NON_BLOCKING == 1

item_t A,B,C;
int wait1 = 1, wait2 = 1, wait3 = 1, wait4 = 1;



void* aba_thread_0(void* arg)
{

  item_t *old = stack->head;
  item_t *new_head = old->next;
  printf("Thread 0 : preempted before cas\n");

  // here should resume main and wait
  wait1 = 0;
  while(wait2);

  cas((size_t*)&(stack->head),(size_t) old,(size_t) new_head);

  printf("Thread 0 : pop \n");

  return 0;
}

void* aba_thread_1(void *arg)
{
  item_t *old = stack->head;
  stack->head = old->next;

  printf("Thread 1 : pop %d -> success\n",  old->value);

  // here wait
  wait3 = 0;
  while(wait4);

  // push A
  A.next = stack->head;
  stack->head = &A;
  printf("Thread 1 : push 1 -> success\n");

  wait2 = 0;
  return 0;
}

void* aba_thread_2(void* arg)
{
  item_t *old = stack->head;
  stack->head = old->next;


  printf("Thread 2 : pop %d -> success\n", old->value);
  wait4 = 0;
  return 0;
}
#endif

int
test_aba()
{

#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  int success, aba_detected = 0;
  // Write here a test for the ABA problem


  // empty the stack
  while(stack->head)
    stack_pop(stack);

  // fill with A,B,C
  A.value = 1;
  B.value = 2;
  C.value = 3;
  A.next = &B;
  B.next = &C;
  C.next = NULL;
  stack->head = &A;

  printf("\n");


  pthread_t threads[ABA_NB_THREADS];
  int i=0;
  
  pthread_create(&threads[0],NULL,aba_thread_0,NULL);
  while(wait1);
  pthread_create(&threads[1],NULL,aba_thread_1,NULL);
  while(wait3);
  pthread_create(&threads[2],NULL,aba_thread_2,NULL);

  for(i=0;i<ABA_NB_THREADS;++i)
  {
    pthread_join(threads[i], NULL);
  }

  // ABA problem detected if stack points to B instead of C
  aba_detected = stack->head == &B;


  success = aba_detected;
  return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  assert(success);
  return success;
#else
  return 1;
#endif
}

int
main(int argc, char **argv)
{

setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];

  test_setup();
  pthread_attr_init(&attr);

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
#if MEASURE == 1
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#else
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
      printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
          start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
          (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
          t_stop[i].tv_nsec);
    }
#endif

  return 0;
}
