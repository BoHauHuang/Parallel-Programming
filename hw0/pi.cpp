#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<thread>
#include<mutex>
#define N 100000000
using namespace std;

int number_in_circle = 0;

int main(){
	srand(time(NULL));
	int i;

	for(i = 0 ; i < N ; i++){
		double x = 2.0*rand()/(RAND_MAX+1.0)-1.0;
		double y = 2.0*rand()/(RAND_MAX+1.0)-1.0;
		double dist = (x*x+y*y);
		if(dist <= 1.0) number_in_circle++;
	}

	double pi_est = 4.0*(double)number_in_circle/((double)N);

	printf("PI: %.3f", pi_est);
	return 0;
}
