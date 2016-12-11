all: 
	gcc milli.c readppm.c CLutilities.c filter.c -lOpenCL -I/usr/local/cuda/include -lGL -lglut -o filter


test:
	gcc milli.c readppm.c CLutilities.c test.c -lOpenCL -I/usr/local/cuda/include -lGL -lglut -o test


init:
	gcc milli.c readppm.c CLutilities.c test_init.c -lOpenCL -I/usr/local/cuda/include -lGL -lglut -o init

clean :
	rm *~ filter test init
	
