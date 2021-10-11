#include<iostream>
#include<iomanip>
#include<chrono>
#include<fstream>

#include<vector>
#include<string>
#include<map>
#include<set>
#include<unordered_map>
#include<unordered_set>

#include<algorithm>
#include<array>
#include<bitset>
#include<cassert>
#include<cstdint>
#include<exception>
#include<functional>
#include<limits>
#include<queue>
#include<regex>
#include<random>

#include <omp.h>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

uint64_t transpose_bitboard_basic(uint64_t b) {
	//引数が8x8 bitboardだとして、転置して返す。

	uint64_t t;

	t = (b ^ (b >> 7)) & 0x00AA00AA00AA00AAULL;
	b = b ^ t ^ (t << 7);
	t = (b ^ (b >> 14)) & 0x0000CCCC0000CCCCULL;
	b = b ^ t ^ (t << 14);
	t = (b ^ (b >> 28)) & 0x00000000F0F0F0F0ULL;
	b = b ^ t ^ (t << 28);

	return b;
}

uint64_t transpose_bitboard_avx2(uint64_t b) {
	//引数が8x8 bitboardだとして、転置して返す。

	const __m256i bb = _mm256_set1_epi64x(b);
	const __m256i x1 = _mm256_sllv_epi64(bb, _mm256_set_epi64x(0, 1, 2, 3));
	const __m256i x2 = _mm256_sllv_epi64(bb, _mm256_set_epi64x(4, 5, 6, 7));
	const int32_t y1 = _mm256_movemask_epi8(x1);
	const int32_t y2 = _mm256_movemask_epi8(x2);

	return (uint64_t(uint32_t(y1)) << 32) + uint64_t(uint32_t(y2));
}

uint64_t transpose_bitboard_sse2(uint64_t b) {
	//引数が8x8 bitboardだとして、転置して返す。

	const __m128i bb = _mm_set1_epi64x(b);
	const __m128i x1 = _mm_sllv_epi64(bb, _mm_set_epi64x(0, 1));
	const __m128i x2 = _mm_sllv_epi64(bb, _mm_set_epi64x(2, 3));
	const __m128i x3 = _mm_sllv_epi64(bb, _mm_set_epi64x(4, 5));
	const __m128i x4 = _mm_sllv_epi64(bb, _mm_set_epi64x(6, 7));
	const int32_t y1 = _mm_movemask_epi8(x1);
	const int32_t y2 = _mm_movemask_epi8(x2);
	const int32_t y3 = _mm_movemask_epi8(x3);
	const int32_t y4 = _mm_movemask_epi8(x4);

	return (uint64_t(uint32_t(y1)) << 48) + (uint64_t(uint32_t(y2)) << 32) + (uint64_t(uint32_t(y3)) << 16) + uint64_t(uint32_t(y4));
}

uint64_t transpose_bitboard_pext(uint64_t x) {
	//引数が8x8 bitboardだとして、転置して返す。

	x = (_pext_u64(x, 0xAAAA'AAAA'AAAA'AAAAULL) << 32) | _pext_u64(x, ~0xAAAA'AAAA'AAAA'AAAAULL);
	x = (_pext_u64(x, 0xAAAA'AAAA'AAAA'AAAAULL) << 32) | _pext_u64(x, ~0xAAAA'AAAA'AAAA'AAAAULL);
	x = (_pext_u64(x, 0xAAAA'AAAA'AAAA'AAAAULL) << 32) | _pext_u64(x, ~0xAAAA'AAAA'AAAA'AAAAULL);

	return x;
}

void unittest(const uint64_t seed, const uint64_t iter) {

	std::mt19937_64 rnd(seed);

	for (int i = 0; i < iter; ++i) {
		const uint64_t a = rnd();
		const uint64_t b1 = transpose_bitboard_basic(a);
		const uint64_t b2 = transpose_bitboard_avx2(a);
		const uint64_t b3 = transpose_bitboard_sse2(a);
		const uint64_t b4 = transpose_bitboard_pext(a);
		assert(b1 == b2 && b1 == b3 && b1 == b4);
	}
}

inline uint64_t xorshift64(uint64_t x) {
	x = x ^ (x << 7);
	return x ^ (x >> 9);
}


#define DEF_BENCH_TRANSPOSE_T(name) \
void bench_transpose_t_##name() {\
	std::cout << "Bench transpose throughput:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		result ^= transpose_bitboard_##name(a);\
		a = xorshift64(a);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\

#define DEF_BENCH_TRANSPOSE_L(name) \
void bench_transpose_l_##name() {\
	std::cout << "Bench transpose latency:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		result ^= transpose_bitboard_##name(a);\
		a = xorshift64(result);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\


DEF_BENCH_TRANSPOSE_T(basic)
DEF_BENCH_TRANSPOSE_T(avx2)
DEF_BENCH_TRANSPOSE_T(sse2)
DEF_BENCH_TRANSPOSE_T(pext)


DEF_BENCH_TRANSPOSE_L(basic)
DEF_BENCH_TRANSPOSE_L(avx2)
DEF_BENCH_TRANSPOSE_L(sse2)
DEF_BENCH_TRANSPOSE_L(pext)

int main() {

	unittest(12345, 100000);

	bench_transpose_t_basic();
	bench_transpose_t_avx2();
	bench_transpose_t_sse2();
	bench_transpose_t_pext();

	bench_transpose_l_basic();
	bench_transpose_l_avx2();
	bench_transpose_l_sse2();
	bench_transpose_l_pext();

	return 0;
}