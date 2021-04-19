#include<iostream>
#include<vector>
#include<string>
#include<cassert>
#include<cstdint>
#include<cstdint>
#include<iomanip>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

uint64_t compress(uint64_t x, uint64_t mask) {
	return _pext_u64(x, mask);
}

uint64_t sag(uint64_t x, uint64_t mask) {
	return (compress(x, mask) << (_mm_popcnt_u64(~mask)))
		| compress(x, ~mask);
}

uint64_t p[6] = {};

void init_p_array(const int x[64]) {

	//引数xは0～63の順列だとする。
	for (int i = 0; i < 64; ++i)
		assert(0 <= x[i] && x[i] < 64);
	for (int i = 0; i < 64; ++i)for (int j = i + 1; j < 64; ++j)
		assert(x[i] != x[j]);

	for (int i = 0; i < 6; ++i)p[i] = 0;

	//2進6桁の値64個からなる配列xを「ビットごとに行列転置」して、
	//2進64桁の値6個の配列pに格納する。
	for (int i = 0; i < 64; ++i) {
		for (uint64_t b = x[i], j = 0; b; ++j, b >>= 1) {
			if (b & 1ULL) {
				p[j] |= 1ULL << i;
			}
		}
	}

	//ハッカーのたのしみ133ページの事前計算
	p[1] = sag(p[1], p[0]);
	p[2] = sag(sag(p[2], p[0]), p[1]);
	p[3] = sag(sag(sag(p[3], p[0]), p[1]), p[2]);
	p[4] = sag(sag(sag(sag(p[4], p[0]), p[1]), p[2]), p[3]);
	p[5] = sag(sag(sag(sag(sag(p[5], p[0]), p[1]), p[2]), p[3]), p[4]);
}

void func(const int x[64], int x_stable[64]) {
	int rep = 0;
	for (int count = 0; count < 64; ++rep) {
		for (int i = 0; i < 64; ++i) {
			if (x[i] == count) {
				x_stable[i] = rep;
				count++;
			}
		}
	}
}

void init_p_array_better(const int x[64]) {

	for (int i = 0; i < 64; ++i)
		assert(0 <= x[i] && x[i] < 64);
	for (int i = 0; i < 64; ++i)for (int j = i + 1; j < 64; ++j)
		assert(x[i] != x[j]);

	int x_stable[64] = {};
	func(x, x_stable);

	for (int i = 0; i < 6; ++i)p[i] = 0;
	for (int i = 0; i < 64; ++i) {
		for (uint64_t b = x_stable[i], j = 0; b; ++j, b >>= 1) {
			if (b & 1ULL) {
				p[j] |= 1ULL << i;
			}
		}
	}

	p[1] = sag(p[1], p[0]);
	p[2] = sag(sag(p[2], p[0]), p[1]);
	p[3] = sag(sag(sag(p[3], p[0]), p[1]), p[2]);
	p[4] = sag(sag(sag(sag(p[4], p[0]), p[1]), p[2]), p[3]);
	p[5] = sag(sag(sag(sag(sag(p[5], p[0]), p[1]), p[2]), p[3]), p[4]);
}

uint64_t permutation(uint64_t x) {

	x = sag(x, p[0]);
	x = sag(x, p[1]);
	x = sag(x, p[2]);
	x = sag(x, p[3]);
	x = sag(x, p[4]);
	x = sag(x, p[5]);

	return x;
}

uint64_t permutation_naive(const uint64_t x, const int s[64]) {

	//引数sは0～63の順列だとする。
	for (int i = 0; i < 64; ++i)
		assert(0 <= s[i] && s[i] < 64);
	for (int i = 0; i < 64; ++i)for (int j = i + 1; j < 64; ++j)
		assert(s[i] != s[j]);

	uint64_t answer = 0;

	for (int i = 0; i < 64; ++i) {
		const uint64_t bit = 1ULL << i;
		if (x & bit)answer |= 1ULL << s[i];
	}

	return answer;
}

int main(void) {

	int x[64];
	for (int i = 0; i < 4; ++i)for (int j = 0; j < 16; ++j) {
		x[i * 16 + j] = (3 - i) * 16 + j;
	}

	init_p_array(x);

	std::cout << "p = (";
	for (int i = 0; i < 6; ++i)std::cout << std::hex << p[i] << (i != 5 ? ", " : ")");
	std::cout << std::endl;

	const uint64_t a = 0x1234'5678'9ABC'DEF0ULL;
	const uint64_t b0 = permutation_naive(a, x);
	const uint64_t b1 = permutation(a);

	init_p_array_better(x);

	std::cout << "p = (";
	for (int i = 0; i < 6; ++i)std::cout << std::hex << p[i] << (i != 5 ? ", " : ")");
	std::cout << std::endl;

	const uint64_t b2 = permutation(a);

	std::cout << "a  = " << a << std::endl;
	std::cout << "b0 = " << b0 << std::endl;
	std::cout << "b1 = " << b1 << std::endl;
	std::cout << "b2 = " << b2 << std::endl;

	return 0;
}
