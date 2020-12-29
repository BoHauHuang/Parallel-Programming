#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<time.h>
#include<pthread.h>
#include<fcntl.h>
#include<immintrin.h>
#include <xmmintrin.h>
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

__m256 ones, RND_MX;
long long int n, number_in_circle;

struct avx_xorshift128plus_key_s{
	__m256i part1;
	__m256i part2;
};

typedef struct avx_xorshift128plus_key_s avx_xorshift128plus_key_t;

static void xorshift128plus_onkeys(uint64_t * ps0, uint64_t * ps1) {
	uint64_t s1 = *ps0;
	const uint64_t s0 = *ps1;
	*ps0 = s0;
	s1 ^= s1 << 23; // a
	*ps1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
}

static void xorshift128plus_jump_onkeys(uint64_t in1, uint64_t in2,
		uint64_t * output1, uint64_t * output2) {
	static const uint64_t JUMP[] = { 0x8a5cd789635d2dff, 0x121fd2155c472f96 };
	uint64_t s0 = 0;
	uint64_t s1 = 0;
	for (unsigned int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++)
		for (int b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b) {
				s0 ^= in1;
				s1 ^= in2;
			}
			xorshift128plus_onkeys(&in1, &in2);
		}
	output1[0] = s0;
	output2[0] = s1;
}

void avx_xorshift128plus_init(uint64_t key1, uint64_t key2,
		avx_xorshift128plus_key_t *key) {
	uint64_t S0[4];
	uint64_t S1[4];
	S0[0] = key1;
	S1[0] = key2;
	xorshift128plus_jump_onkeys(*S0, *S1, S0 + 1, S1 + 1);
	xorshift128plus_jump_onkeys(*(S0 + 1), *(S1 + 1), S0 + 2, S1 + 2);
	xorshift128plus_jump_onkeys(*(S0 + 2), *(S1 + 2), S0 + 3, S1 + 3);
	key->part1 = _mm256_loadu_si256((const __m256i *) S0);
	key->part2 = _mm256_loadu_si256((const __m256i *) S1);
}

__m256i avx_xorshift128plus(avx_xorshift128plus_key_t *key) {
	__m256i s1 = key->part1;
	const __m256i s0 = key->part2;
	key->part1 = key->part2;
	s1 = _mm256_xor_si256(key->part2, _mm256_slli_epi64(key->part2, 23));
	key->part2 = _mm256_xor_si256(
			_mm256_xor_si256(_mm256_xor_si256(s1, s0),
					_mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));
	return _mm256_add_epi64(key->part2, s0);
}

static inline void* calc(void* arg){
	avx_xorshift128plus_key_t xkey, ykey;
	
	avx_xorshift128plus_init(123+(uint64_t)arg, 123+(uint64_t)arg, &xkey);
	avx_xorshift128plus_init(456+(uint64_t)arg, 456+(uint64_t)arg, &ykey);
	__m256i numx, numy;
	__m256 xx, yy, mx, my, mask, add, dx, dy, vcnt;
	vcnt = _mm256_setzero_ps();
	long long int i, cnt = 0;
	short j;
	float *ans = (float*) _mm_malloc(8*sizeof(float), 32);
	
	for(i=0;i<n;i+=8){
		numx = avx_xorshift128plus(&xkey);
		xx = _mm256_cvtepi32_ps(numx);
		numy = avx_xorshift128plus(&ykey);
		yy = _mm256_cvtepi32_ps(numy);

		dx = _mm256_div_ps(xx, RND_MX);
		dy = _mm256_div_ps(yy, RND_MX);	

		mx = _mm256_mul_ps(dx,dx);
		my = _mm256_mul_ps(dy,dy);

		xx = _mm256_add_ps(mx,my);

		mask = _mm256_cmp_ps(xx, ones, _CMP_LE_OQ);
		add = _mm256_and_ps(mask, ones);
		vcnt = _mm256_add_ps(vcnt, add);
		
		if(i%1500000000 == 0){
			j = 8;
			_mm256_store_ps(ans, vcnt);
			vcnt = _mm256_setzero_ps();
		
			for( ;j; ){
				--j;
				if(ans[j]) cnt+=ans[j];
			}	
		}
	}
	_mm256_store_ps(ans, vcnt);
	j = 8;
	for( ;j; ){
		--j;
		if(ans[j]) cnt+=ans[j];
	}
	pthread_mutex_lock(&mtx);
	//_mm256_add_pd(number_in_circle, cnt);
	number_in_circle += cnt;
	pthread_mutex_unlock(&mtx);
	pthread_exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[]){
	
	long long int N, M, i;
	M = atoll(argv[1]);
	N = atoll(argv[2]);
	
	//number_in_circle = _mm256_setzero_pd();
	
	RND_MX = _mm256_set1_ps(INT32_MAX);
	ones = _mm256_set1_ps(1.0f);

	pthread_t t[M];
	n = N/M;
	i = M;
	for( ;i; ){
		--i;
		pthread_create(&t[i], NULL, calc, (void*)i);
	}
	for( ;M; ){
		--M;
		pthread_join(t[M], NULL);
	}
	
	//double output[5];
	//output[4] = 0.0;

	//_mm256_store_pd(output, number_in_circle);
	//i = 4;
	//for( ;i; ){
	//	--i;
	//	output[4] += output[i];
	//}
	double cn = (double)number_in_circle/N;
	double pi_est = 4.0*cn;
	printf("%.6lf\n", pi_est);
	return 0;
}
