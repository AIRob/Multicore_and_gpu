#include <stdlib.h> 

#include <drake/schedule.h> 

#include <drake/platform.h> 

#define TASK_NAME inner_9
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_inner_9 1

#define TASK_NAME inner_10
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_inner_10 1

#define TASK_NAME inner_11
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_inner_11 1

#define TASK_NAME inner_12
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_inner_12 1

#define TASK_NAME inner_13
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_inner_13 1

#define TASK_NAME inner_14
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_inner_14 1

#define TASK_NAME leaf_1
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_1 1

#define TASK_NAME leaf_2
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_2 1

#define TASK_NAME leaf_3
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_3 1

#define TASK_NAME leaf_4
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_4 1

#define TASK_NAME leaf_5
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_5 1

#define TASK_NAME leaf_6
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_6 1

#define TASK_NAME leaf_7
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_7 1

#define TASK_NAME leaf_8
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_8 1

#define TASK_NAME root
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_root 1

int drake_task_number()
{
	return 15;
}

char* drake_task_name(size_t index)
{
	switch(index - 1)
	{
		case 0:
			return "inner_9";
		break;
		case 1:
			return "inner_10";
		break;
		case 2:
			return "inner_11";
		break;
		case 3:
			return "inner_12";
		break;
		case 4:
			return "inner_13";
		break;
		case 5:
			return "inner_14";
		break;
		case 6:
			return "leaf_1";
		break;
		case 7:
			return "leaf_2";
		break;
		case 8:
			return "leaf_3";
		break;
		case 9:
			return "leaf_4";
		break;
		case 10:
			return "leaf_5";
		break;
		case 11:
			return "leaf_6";
		break;
		case 12:
			return "leaf_7";
		break;
		case 13:
			return "leaf_8";
		break;
		case 14:
			return "root";
		break;
		default:
			return "invalid task id";
		break;
	}
}

void drake_schedule_init(drake_schedule_t* schedule)
{
	schedule->core_number = 4;
	schedule->task_number = 15;
	schedule->stage_time = 0;

	schedule->tasks_in_core = malloc(sizeof(size_t) * schedule->core_number);

	schedule->task_name = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->task_name[0] = "inner_9";
	schedule->task_name[1] = "inner_10";
	schedule->task_name[2] = "inner_11";
	schedule->task_name[3] = "inner_12";
	schedule->task_name[4] = "inner_13";
	schedule->task_name[5] = "inner_14";
	schedule->task_name[6] = "leaf_1";
	schedule->task_name[7] = "leaf_2";
	schedule->task_name[8] = "leaf_3";
	schedule->task_name[9] = "leaf_4";
	schedule->task_name[10] = "leaf_5";
	schedule->task_name[11] = "leaf_6";
	schedule->task_name[12] = "leaf_7";
	schedule->task_name[13] = "leaf_8";
	schedule->task_name[14] = "root";


	schedule->task_workload = malloc(sizeof(double) * schedule->task_number);
	schedule->task_workload[0] = 1;
	schedule->task_workload[1] = 1;
	schedule->task_workload[2] = 1;
	schedule->task_workload[3] = 1;
	schedule->task_workload[4] = 1;
	schedule->task_workload[5] = 1;
	schedule->task_workload[6] = 1;
	schedule->task_workload[7] = 1;
	schedule->task_workload[8] = 1;
	schedule->task_workload[9] = 1;
	schedule->task_workload[10] = 1;
	schedule->task_workload[11] = 1;
	schedule->task_workload[12] = 1;
	schedule->task_workload[13] = 1;
	schedule->task_workload[14] = 1;

	schedule->tasks_in_core[0] = 4;
	schedule->tasks_in_core[1] = 4;
	schedule->tasks_in_core[2] = 4;
	schedule->tasks_in_core[3] = 3;

	schedule->consumers_in_core = malloc(sizeof(size_t) * schedule->core_number);
	schedule->consumers_in_core[0] = 2;
	schedule->consumers_in_core[1] = 4;
	schedule->consumers_in_core[2] = 3;
	schedule->consumers_in_core[3] = 2;

	schedule->producers_in_core = malloc(sizeof(size_t) * schedule->core_number);
	schedule->producers_in_core[0] = 2;
	schedule->producers_in_core[1] = 4;
	schedule->producers_in_core[2] = 4;
	schedule->producers_in_core[3] = 1;

	schedule->consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->consumers_in_task[0] = 1;
	schedule->consumers_in_task[1] = 1;
	schedule->consumers_in_task[2] = 1;
	schedule->consumers_in_task[3] = 1;
	schedule->consumers_in_task[4] = 1;
	schedule->consumers_in_task[5] = 1;
	schedule->consumers_in_task[6] = 1;
	schedule->consumers_in_task[7] = 1;
	schedule->consumers_in_task[8] = 1;
	schedule->consumers_in_task[9] = 1;
	schedule->consumers_in_task[10] = 1;
	schedule->consumers_in_task[11] = 1;
	schedule->consumers_in_task[12] = 1;
	schedule->consumers_in_task[13] = 1;
	schedule->consumers_in_task[14] = 0;

	schedule->producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->producers_in_task[0] = 2;
	schedule->producers_in_task[1] = 2;
	schedule->producers_in_task[2] = 2;
	schedule->producers_in_task[3] = 2;
	schedule->producers_in_task[4] = 2;
	schedule->producers_in_task[5] = 2;
	schedule->producers_in_task[6] = 0;
	schedule->producers_in_task[7] = 0;
	schedule->producers_in_task[8] = 0;
	schedule->producers_in_task[9] = 0;
	schedule->producers_in_task[10] = 0;
	schedule->producers_in_task[11] = 0;
	schedule->producers_in_task[12] = 0;
	schedule->producers_in_task[13] = 0;
	schedule->producers_in_task[14] = 2;

	schedule->remote_consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->remote_consumers_in_task[0] = 0;
	schedule->remote_consumers_in_task[1] = 1;
	schedule->remote_consumers_in_task[2] = 1;
	schedule->remote_consumers_in_task[3] = 1;
	schedule->remote_consumers_in_task[4] = 1;
	schedule->remote_consumers_in_task[5] = 1;
	schedule->remote_consumers_in_task[6] = 0;
	schedule->remote_consumers_in_task[7] = 1;
	schedule->remote_consumers_in_task[8] = 1;
	schedule->remote_consumers_in_task[9] = 1;
	schedule->remote_consumers_in_task[10] = 1;
	schedule->remote_consumers_in_task[11] = 1;
	schedule->remote_consumers_in_task[12] = 1;
	schedule->remote_consumers_in_task[13] = 0;
	schedule->remote_consumers_in_task[14] = 0;

	schedule->remote_producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->remote_producers_in_task[0] = 1;
	schedule->remote_producers_in_task[1] = 2;
	schedule->remote_producers_in_task[2] = 2;
	schedule->remote_producers_in_task[3] = 1;
	schedule->remote_producers_in_task[4] = 1;
	schedule->remote_producers_in_task[5] = 2;
	schedule->remote_producers_in_task[6] = 0;
	schedule->remote_producers_in_task[7] = 0;
	schedule->remote_producers_in_task[8] = 0;
	schedule->remote_producers_in_task[9] = 0;
	schedule->remote_producers_in_task[10] = 0;
	schedule->remote_producers_in_task[11] = 0;
	schedule->remote_producers_in_task[12] = 0;
	schedule->remote_producers_in_task[13] = 0;
	schedule->remote_producers_in_task[14] = 2;

	schedule->producers_id = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->producers_rate = malloc(sizeof(size_t**) * schedule->task_number);
	schedule->producers_name = malloc(sizeof(char*) * schedule->task_number);
	schedule->producers_id[0] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[0][0] = 7;
	schedule->producers_id[0][1] = 8;
	schedule->producers_rate[0] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[0][0] = 1;
	schedule->producers_rate[0][1] = 1;
	schedule->producers_name[0] = malloc(sizeof(char*) * 2);
	schedule->producers_name[0][0] = "output";
	schedule->producers_name[0][1] = "output";
	schedule->producers_id[1] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[1][0] = 9;
	schedule->producers_id[1][1] = 10;
	schedule->producers_rate[1] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[1][0] = 1;
	schedule->producers_rate[1][1] = 1;
	schedule->producers_name[1] = malloc(sizeof(char*) * 2);
	schedule->producers_name[1][0] = "output";
	schedule->producers_name[1][1] = "output";
	schedule->producers_id[2] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[2][0] = 11;
	schedule->producers_id[2][1] = 12;
	schedule->producers_rate[2] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[2][0] = 1;
	schedule->producers_rate[2][1] = 1;
	schedule->producers_name[2] = malloc(sizeof(char*) * 2);
	schedule->producers_name[2][0] = "output";
	schedule->producers_name[2][1] = "output";
	schedule->producers_id[3] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[3][0] = 13;
	schedule->producers_id[3][1] = 14;
	schedule->producers_rate[3] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[3][0] = 1;
	schedule->producers_rate[3][1] = 1;
	schedule->producers_name[3] = malloc(sizeof(char*) * 2);
	schedule->producers_name[3][0] = "output";
	schedule->producers_name[3][1] = "output";
	schedule->producers_id[4] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[4][0] = 1;
	schedule->producers_id[4][1] = 2;
	schedule->producers_rate[4] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[4][0] = 1;
	schedule->producers_rate[4][1] = 1;
	schedule->producers_name[4] = malloc(sizeof(char*) * 2);
	schedule->producers_name[4][0] = "output";
	schedule->producers_name[4][1] = "output";
	schedule->producers_id[5] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[5][0] = 3;
	schedule->producers_id[5][1] = 4;
	schedule->producers_rate[5] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[5][0] = 1;
	schedule->producers_rate[5][1] = 1;
	schedule->producers_name[5] = malloc(sizeof(char*) * 2);
	schedule->producers_name[5][0] = "output";
	schedule->producers_name[5][1] = "output";
	schedule->producers_id[6] = NULL;
	schedule->producers_rate[6] = NULL;
	schedule->producers_name[6] = NULL;
	schedule->producers_id[7] = NULL;
	schedule->producers_rate[7] = NULL;
	schedule->producers_name[7] = NULL;
	schedule->producers_id[8] = NULL;
	schedule->producers_rate[8] = NULL;
	schedule->producers_name[8] = NULL;
	schedule->producers_id[9] = NULL;
	schedule->producers_rate[9] = NULL;
	schedule->producers_name[9] = NULL;
	schedule->producers_id[10] = NULL;
	schedule->producers_rate[10] = NULL;
	schedule->producers_name[10] = NULL;
	schedule->producers_id[11] = NULL;
	schedule->producers_rate[11] = NULL;
	schedule->producers_name[11] = NULL;
	schedule->producers_id[12] = NULL;
	schedule->producers_rate[12] = NULL;
	schedule->producers_name[12] = NULL;
	schedule->producers_id[13] = NULL;
	schedule->producers_rate[13] = NULL;
	schedule->producers_name[13] = NULL;
	schedule->producers_id[14] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[14][0] = 5;
	schedule->producers_id[14][1] = 6;
	schedule->producers_rate[14] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[14][0] = 1;
	schedule->producers_rate[14][1] = 1;
	schedule->producers_name[14] = malloc(sizeof(char*) * 2);
	schedule->producers_name[14][0] = "output";
	schedule->producers_name[14][1] = "output";

	schedule->consumers_id = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->consumers_rate = malloc(sizeof(size_t**) * schedule->task_number);
	schedule->consumers_name = malloc(sizeof(char*) * schedule->task_number);
	schedule->consumers_id[0] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[0][0] = 5;
	schedule->consumers_rate[0] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[0][0] = 1;
	schedule->consumers_name[0] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[0][0] = "left";
	schedule->consumers_id[1] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[1][0] = 5;
	schedule->consumers_rate[1] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[1][0] = 1;
	schedule->consumers_name[1] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[1][0] = "right";
	schedule->consumers_id[2] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[2][0] = 6;
	schedule->consumers_rate[2] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[2][0] = 1;
	schedule->consumers_name[2] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[2][0] = "left";
	schedule->consumers_id[3] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[3][0] = 6;
	schedule->consumers_rate[3] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[3][0] = 1;
	schedule->consumers_name[3] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[3][0] = "right";
	schedule->consumers_id[4] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[4][0] = 15;
	schedule->consumers_rate[4] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[4][0] = 1;
	schedule->consumers_name[4] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[4][0] = "left";
	schedule->consumers_id[5] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[5][0] = 15;
	schedule->consumers_rate[5] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[5][0] = 1;
	schedule->consumers_name[5] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[5][0] = "right";
	schedule->consumers_id[6] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[6][0] = 1;
	schedule->consumers_rate[6] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[6][0] = 1;
	schedule->consumers_name[6] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[6][0] = "left";
	schedule->consumers_id[7] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[7][0] = 1;
	schedule->consumers_rate[7] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[7][0] = 1;
	schedule->consumers_name[7] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[7][0] = "right";
	schedule->consumers_id[8] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[8][0] = 2;
	schedule->consumers_rate[8] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[8][0] = 1;
	schedule->consumers_name[8] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[8][0] = "left";
	schedule->consumers_id[9] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[9][0] = 2;
	schedule->consumers_rate[9] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[9][0] = 1;
	schedule->consumers_name[9] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[9][0] = "right";
	schedule->consumers_id[10] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[10][0] = 3;
	schedule->consumers_rate[10] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[10][0] = 1;
	schedule->consumers_name[10] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[10][0] = "left";
	schedule->consumers_id[11] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[11][0] = 3;
	schedule->consumers_rate[11] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[11][0] = 1;
	schedule->consumers_name[11] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[11][0] = "right";
	schedule->consumers_id[12] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[12][0] = 4;
	schedule->consumers_rate[12] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[12][0] = 1;
	schedule->consumers_name[12] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[12][0] = "left";
	schedule->consumers_id[13] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[13][0] = 4;
	schedule->consumers_rate[13] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[13][0] = 1;
	schedule->consumers_name[13] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[13][0] = "right";
	schedule->consumers_id[14] = NULL;
	schedule->consumers_rate[14] = NULL;
	schedule->consumers_name[14] = NULL;

	schedule->schedule = malloc(sizeof(drake_schedule_task_t*) * schedule->core_number);
	schedule->schedule[0] = malloc(sizeof(drake_schedule_task_t) * 4);
	schedule->schedule[0][0].id = 11;
	schedule->schedule[0][0].start_time = 0;
	schedule->schedule[0][0].frequency = 16;
	schedule->schedule[0][1].id = 7;
	schedule->schedule[0][1].start_time = 2.86e-07;
	schedule->schedule[0][1].frequency = 16;
	schedule->schedule[0][2].id = 5;
	schedule->schedule[0][2].start_time = 5.71e-07;
	schedule->schedule[0][2].frequency = 16;
	schedule->schedule[0][3].id = 1;
	schedule->schedule[0][3].start_time = 8.57e-07;
	schedule->schedule[0][3].frequency = 16;
	schedule->schedule[1] = malloc(sizeof(drake_schedule_task_t) * 4);
	schedule->schedule[1][0].id = 12;
	schedule->schedule[1][0].start_time = 0;
	schedule->schedule[1][0].frequency = 16;
	schedule->schedule[1][1].id = 8;
	schedule->schedule[1][1].start_time = 2.86e-07;
	schedule->schedule[1][1].frequency = 16;
	schedule->schedule[1][2].id = 6;
	schedule->schedule[1][2].start_time = 5.71e-07;
	schedule->schedule[1][2].frequency = 16;
	schedule->schedule[1][3].id = 2;
	schedule->schedule[1][3].start_time = 8.57e-07;
	schedule->schedule[1][3].frequency = 16;
	schedule->schedule[2] = malloc(sizeof(drake_schedule_task_t) * 4);
	schedule->schedule[2][0].id = 13;
	schedule->schedule[2][0].start_time = 0;
	schedule->schedule[2][0].frequency = 16;
	schedule->schedule[2][1].id = 9;
	schedule->schedule[2][1].start_time = 2.86e-07;
	schedule->schedule[2][1].frequency = 16;
	schedule->schedule[2][2].id = 15;
	schedule->schedule[2][2].start_time = 5.71e-07;
	schedule->schedule[2][2].frequency = 16;
	schedule->schedule[2][3].id = 3;
	schedule->schedule[2][3].start_time = 8.57e-07;
	schedule->schedule[2][3].frequency = 16;
	schedule->schedule[3] = malloc(sizeof(drake_schedule_task_t) * 3);
	schedule->schedule[3][0].id = 14;
	schedule->schedule[3][0].start_time = 0;
	schedule->schedule[3][0].frequency = 16;
	schedule->schedule[3][1].id = 10;
	schedule->schedule[3][1].start_time = 2.86e-07;
	schedule->schedule[3][1].frequency = 16;
	schedule->schedule[3][2].id = 4;
	schedule->schedule[3][2].start_time = 5.71e-07;
	schedule->schedule[3][2].frequency = 16;
}

void drake_schedule_destroy(drake_schedule_t* schedule)
{
	free(schedule->schedule[0]);
	free(schedule->schedule[1]);
	free(schedule->schedule[2]);
	free(schedule->schedule[3]);

	free(schedule->schedule);
	free(schedule->consumers_id[0]);
	if(schedule->consumers_rate[0] != NULL)
	{
		free(schedule->consumers_rate[0]);
	}
	free(schedule->consumers_name[0]);
	free(schedule->consumers_id[1]);
	if(schedule->consumers_rate[1] != NULL)
	{
		free(schedule->consumers_rate[1]);
	}
	free(schedule->consumers_name[1]);
	free(schedule->consumers_id[2]);
	if(schedule->consumers_rate[2] != NULL)
	{
		free(schedule->consumers_rate[2]);
	}
	free(schedule->consumers_name[2]);
	free(schedule->consumers_id[3]);
	if(schedule->consumers_rate[3] != NULL)
	{
		free(schedule->consumers_rate[3]);
	}
	free(schedule->consumers_name[3]);
	free(schedule->consumers_id[4]);
	if(schedule->consumers_rate[4] != NULL)
	{
		free(schedule->consumers_rate[4]);
	}
	free(schedule->consumers_name[4]);
	free(schedule->consumers_id[5]);
	if(schedule->consumers_rate[5] != NULL)
	{
		free(schedule->consumers_rate[5]);
	}
	free(schedule->consumers_name[5]);
	free(schedule->consumers_id[6]);
	if(schedule->consumers_rate[6] != NULL)
	{
		free(schedule->consumers_rate[6]);
	}
	free(schedule->consumers_name[6]);
	free(schedule->consumers_id[7]);
	if(schedule->consumers_rate[7] != NULL)
	{
		free(schedule->consumers_rate[7]);
	}
	free(schedule->consumers_name[7]);
	free(schedule->consumers_id[8]);
	if(schedule->consumers_rate[8] != NULL)
	{
		free(schedule->consumers_rate[8]);
	}
	free(schedule->consumers_name[8]);
	free(schedule->consumers_id[9]);
	if(schedule->consumers_rate[9] != NULL)
	{
		free(schedule->consumers_rate[9]);
	}
	free(schedule->consumers_name[9]);
	free(schedule->consumers_id[10]);
	if(schedule->consumers_rate[10] != NULL)
	{
		free(schedule->consumers_rate[10]);
	}
	free(schedule->consumers_name[10]);
	free(schedule->consumers_id[11]);
	if(schedule->consumers_rate[11] != NULL)
	{
		free(schedule->consumers_rate[11]);
	}
	free(schedule->consumers_name[11]);
	free(schedule->consumers_id[12]);
	if(schedule->consumers_rate[12] != NULL)
	{
		free(schedule->consumers_rate[12]);
	}
	free(schedule->consumers_name[12]);
	free(schedule->consumers_id[13]);
	if(schedule->consumers_rate[13] != NULL)
	{
		free(schedule->consumers_rate[13]);
	}
	free(schedule->consumers_name[13]);
	free(schedule->consumers_id[14]);
	if(schedule->consumers_rate[14] != NULL)
	{
		free(schedule->consumers_rate[14]);
	}
	free(schedule->consumers_name[14]);
	free(schedule->consumers_id);
	free(schedule->consumers_rate);
	free(schedule->consumers_name);

	free(schedule->producers_id[0]);
	if(schedule->producers_rate[0] != NULL)
	{
		free(schedule->producers_rate[0]);
	}
	free(schedule->producers_name[0]);
	free(schedule->producers_id[1]);
	if(schedule->producers_rate[1] != NULL)
	{
		free(schedule->producers_rate[1]);
	}
	free(schedule->producers_name[1]);
	free(schedule->producers_id[2]);
	if(schedule->producers_rate[2] != NULL)
	{
		free(schedule->producers_rate[2]);
	}
	free(schedule->producers_name[2]);
	free(schedule->producers_id[3]);
	if(schedule->producers_rate[3] != NULL)
	{
		free(schedule->producers_rate[3]);
	}
	free(schedule->producers_name[3]);
	free(schedule->producers_id[4]);
	if(schedule->producers_rate[4] != NULL)
	{
		free(schedule->producers_rate[4]);
	}
	free(schedule->producers_name[4]);
	free(schedule->producers_id[5]);
	if(schedule->producers_rate[5] != NULL)
	{
		free(schedule->producers_rate[5]);
	}
	free(schedule->producers_name[5]);
	free(schedule->producers_id[6]);
	if(schedule->producers_rate[6] != NULL)
	{
		free(schedule->producers_rate[6]);
	}
	free(schedule->producers_name[6]);
	free(schedule->producers_id[7]);
	if(schedule->producers_rate[7] != NULL)
	{
		free(schedule->producers_rate[7]);
	}
	free(schedule->producers_name[7]);
	free(schedule->producers_id[8]);
	if(schedule->producers_rate[8] != NULL)
	{
		free(schedule->producers_rate[8]);
	}
	free(schedule->producers_name[8]);
	free(schedule->producers_id[9]);
	if(schedule->producers_rate[9] != NULL)
	{
		free(schedule->producers_rate[9]);
	}
	free(schedule->producers_name[9]);
	free(schedule->producers_id[10]);
	if(schedule->producers_rate[10] != NULL)
	{
		free(schedule->producers_rate[10]);
	}
	free(schedule->producers_name[10]);
	free(schedule->producers_id[11]);
	if(schedule->producers_rate[11] != NULL)
	{
		free(schedule->producers_rate[11]);
	}
	free(schedule->producers_name[11]);
	free(schedule->producers_id[12]);
	if(schedule->producers_rate[12] != NULL)
	{
		free(schedule->producers_rate[12]);
	}
	free(schedule->producers_name[12]);
	free(schedule->producers_id[13]);
	if(schedule->producers_rate[13] != NULL)
	{
		free(schedule->producers_rate[13]);
	}
	free(schedule->producers_name[13]);
	free(schedule->producers_id[14]);
	if(schedule->producers_rate[14] != NULL)
	{
		free(schedule->producers_rate[14]);
	}
	free(schedule->producers_name[14]);
	free(schedule->producers_id);
	free(schedule->producers_rate);
	free(schedule->producers_name);
	free(schedule->task_workload);
	free(schedule->remote_producers_in_task);
	free(schedule->remote_consumers_in_task);
	free(schedule->producers_in_task);
	free(schedule->consumers_in_task);
	free(schedule->producers_in_core);
	free(schedule->consumers_in_core);
	free(schedule->tasks_in_core);
	free(schedule->task_name);
}

size_t
drake_task_width(task_tp task){
	size_t task_width[15] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, };
	return task_width[task->id - 1];
}

size_t
drake_core_id(task_tp task){
	size_t local_core_id[4][15] = {
			{0, 4, 4, 4, 0, 4, 0, 4, 4, 4, 0, 4, 4, 4, 4, },
			{4, 0, 4, 4, 4, 0, 4, 0, 4, 4, 4, 0, 4, 4, 4, },
			{4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 0, 4, 0, },
			{4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 0, 4, },
		};
	return local_core_id[drake_platform_core_id()][task->id - 1];
}

void*
drake_function(size_t id, task_status_t status)
{
	switch(id)
	{
		default:
			// TODO: Raise an alert
		break;
		case 1:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, inner_9);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, inner_9);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, inner_9);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, inner_9);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, inner_9);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 2:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, inner_10);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, inner_10);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, inner_10);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, inner_10);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, inner_10);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 3:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, inner_11);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, inner_11);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, inner_11);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, inner_11);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, inner_11);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 4:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, inner_12);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, inner_12);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, inner_12);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, inner_12);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, inner_12);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 5:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, inner_13);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, inner_13);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, inner_13);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, inner_13);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, inner_13);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 6:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, inner_14);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, inner_14);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, inner_14);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, inner_14);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, inner_14);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 7:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_1);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_1);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_1);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_1);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_1);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 8:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_2);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_2);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_2);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_2);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_2);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 9:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_3);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_3);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_3);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_3);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_3);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 10:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_4);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_4);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_4);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_4);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_4);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 11:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_5);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_5);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_5);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_5);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_5);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 12:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_6);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_6);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_6);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_6);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_6);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 13:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_7);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_7);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_7);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_7);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_7);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 14:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_8);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_8);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_8);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_8);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_8);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 15:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, root);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, root);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, root);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, root);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, root);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;

	}

	return 0;
}
