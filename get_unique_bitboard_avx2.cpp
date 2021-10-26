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

//#include <omp.h>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

uint64_t vertical_mirror(uint64_t b) {

	b = ((b >> 8) & 0x00FF00FF00FF00FFULL) | ((b << 8) & 0xFF00FF00FF00FF00ULL);
	b = ((b >> 16) & 0x0000FFFF0000FFFFULL) | ((b << 16) & 0xFFFF0000FFFF0000ULL);
	b = ((b >> 32) & 0x00000000FFFFFFFFULL) | ((b << 32) & 0xFFFFFFFF00000000ULL);

	return b;
}

uint64_t horizontal_mirror(uint64_t b) {

	b = ((b >> 1) & 0x5555555555555555ULL) | ((b << 1) & 0xAAAAAAAAAAAAAAAAULL);
	b = ((b >> 2) & 0x3333333333333333ULL) | ((b << 2) & 0xCCCCCCCCCCCCCCCCULL);
	b = ((b >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((b << 4) & 0xF0F0F0F0F0F0F0F0ULL);

	return b;
}

uint64_t transpose(uint64_t b) {

	uint64_t t;

	t = (b ^ (b >> 7)) & 0x00AA00AA00AA00AAULL;
	b = b ^ t ^ (t << 7);
	t = (b ^ (b >> 14)) & 0x0000CCCC0000CCCCULL;
	b = b ^ t ^ (t << 14);
	t = (b ^ (b >> 28)) & 0x00000000F0F0F0F0ULL;
	b = b ^ t ^ (t << 28);

	return b;
}

uint64_t symmetry_naive(const uint32_t s, uint64_t b) {

	uint64_t answer = b;

	if (s & 1)answer = horizontal_mirror(answer);
	if (s & 2)answer = vertical_mirror(answer);
	if (s & 4)answer = transpose(answer);

	return answer;
}

uint64_t get_unique_naive(const uint64_t b) {

	uint64_t answer = b;

	for (uint32_t i = 1; i <= 7; ++i) {
		const uint64_t new_code = symmetry_naive(i, b);
		answer = std::min(answer, new_code);
	}

	return answer;
}

uint64_t get_unique_avx2_1(const uint64_t b) {

	constexpr uint64_t FULL = 0xFFFF'FFFF'FFFF'FFFFULL;

	const __m256i bb0 = _mm256_set1_epi64x(int64_t(b));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 8, 1, 0)), _mm256_set_epi64x(FULL, 0x00FF00FF00FF00FFLL, 0x5555555555555555LL, FULL));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 8, 1, 0)), _mm256_set_epi64x(FULL, 0xFF00FF00FF00FF00LL, 0xAAAAAAAAAAAAAAAALL, FULL));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 16, 2, 0)), _mm256_set_epi64x(FULL, 0x0000FFFF0000FFFFLL, 0x3333333333333333LL, FULL));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 16, 2, 0)), _mm256_set_epi64x(FULL, 0xFFFF0000FFFF0000LL, 0xCCCCCCCCCCCCCCCCLL, FULL));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(tt2, _mm256_set_epi64x(0, 32, 4, 0)), _mm256_set_epi64x(FULL, 0x00000000FFFFFFFFLL, 0x0F0F0F0F0F0F0F0FLL, FULL));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 32, 4, 0)), _mm256_set_epi64x(FULL, 0xFFFFFFFF00000000LL, 0xF0F0F0F0F0F0F0F0LL, FULL));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	constexpr auto f = [](const uint8_t i) {
		return uint8_t(((i & 1) << 3) + ((i & 2) << 1) + ((i & 4) >> 1) + ((i & 8) >> 3));
	};

	const __m128i rvr1 = _mm_set_epi8(f(15), f(14), f(13), f(12), f(11), f(10), f(9), f(8), f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0));

	const __m128i rva1 = _mm_set_epi64x((b >> 4) & 0x0F0F'0F0F'0F0F'0F0FULL, b & 0x0F0F'0F0F'0F0F'0F0FULL);
	const __m128i rva2 = _mm_shuffle_epi8(rvr1, rva1);
	const __m128i rva3 = _mm_shuffle_epi8(rva2, _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7));
	const __m128i rva4 = _mm_shuffle_epi32(rva3, 0b00001110);
	const __m128i rva5 = _mm_add_epi32(rva4, _mm_slli_epi64(rva3, 4));

	const __m256i tt = _mm256_blend_epi32(tt3, _mm256_zextsi128_si256(rva5), 0b00000011);

	const __m256i x1 = _mm256_and_si256(_mm256_xor_si256(tt, _mm256_srli_epi64(tt, 7)), _mm256_set1_epi64x(0x00AA00AA00AA00AALL));
	const __m256i y1 = _mm256_xor_si256(tt, _mm256_xor_si256(x1, _mm256_slli_epi64(x1, 7)));
	const __m256i x2 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64(y1, 14)), _mm256_set1_epi64x(0x0000CCCC0000CCCCLL));
	const __m256i y2 = _mm256_xor_si256(y1, _mm256_xor_si256(x2, _mm256_slli_epi64(x2, 14)));
	const __m256i x3 = _mm256_and_si256(_mm256_xor_si256(y2, _mm256_srli_epi64(y2, 28)), _mm256_set1_epi64x(0x00000000F0F0F0F0LL));
	const __m256i zz = _mm256_xor_si256(y2, _mm256_xor_si256(x3, _mm256_slli_epi64(x3, 28)));

	const __m256i a1 = _mm256_sub_epi64(zz, tt);
	const __m256i a2 = _mm256_srai_epi32(a1, 32);
	const __m256i a3 = _mm256_shuffle_epi32(a2, 0b11110101);
	const __m256i a4 = _mm256_blendv_epi8(tt, zz, a3);

	alignas(32) uint64_t result[4] = {};
	_mm256_storeu_si256((__m256i*)result, a4);

	result[0] = std::min(result[0], result[1]);
	result[0] = std::min(result[0], result[2]);
	result[0] = std::min(result[0], result[3]);
	return result[0];
}

uint64_t get_unique_avx2_2(const uint64_t b) {

	uint64_t v = b;

	v = ((v >> 8) & 0x00FF00FF00FF00FFULL) | ((v << 8) & 0xFF00FF00FF00FF00ULL);
	v = ((v >> 16) & 0x0000FFFF0000FFFFULL) | ((v << 16) & 0xFFFF0000FFFF0000ULL);
	v = ((v >> 32) & 0x00000000FFFFFFFFULL) | ((v << 32) & 0xFFFFFFFF00000000ULL);

	constexpr uint64_t FULL = 0xFFFF'FFFF'FFFF'FFFFULL;

	const __m256i bb0 = _mm256_set_epi64x(int64_t(b), int64_t(v), int64_t(b), int64_t(v));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, 0x5555555555555555LL, 0x5555555555555555LL));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, 0xAAAAAAAAAAAAAAAALL, 0xAAAAAAAAAAAAAAAALL));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, 0x3333333333333333LL, 0x3333333333333333LL));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, 0xCCCCCCCCCCCCCCCCLL, 0xCCCCCCCCCCCCCCCCLL));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(tt2, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, 0x0F0F0F0F0F0F0F0FLL, 0x0F0F0F0F0F0F0F0FLL));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, 0xF0F0F0F0F0F0F0F0LL, 0xF0F0F0F0F0F0F0F0LL));
	const __m256i tt = _mm256_or_si256(tt3lo, tt3hi);

	const __m256i x1 = _mm256_and_si256(_mm256_xor_si256(tt, _mm256_srli_epi64(tt, 7)), _mm256_set1_epi64x(0x00AA00AA00AA00AALL));
	const __m256i y1 = _mm256_xor_si256(tt, _mm256_xor_si256(x1, _mm256_slli_epi64(x1, 7)));
	const __m256i x2 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64(y1, 14)), _mm256_set1_epi64x(0x0000CCCC0000CCCCLL));
	const __m256i y2 = _mm256_xor_si256(y1, _mm256_xor_si256(x2, _mm256_slli_epi64(x2, 14)));
	const __m256i x3 = _mm256_and_si256(_mm256_xor_si256(y2, _mm256_srli_epi64(y2, 28)), _mm256_set1_epi64x(0x00000000F0F0F0F0LL));
	const __m256i zz = _mm256_xor_si256(y2, _mm256_xor_si256(x3, _mm256_slli_epi64(x3, 28)));

	const __m256i a1 = _mm256_sub_epi64(zz, tt);
	const __m256i a2 = _mm256_srai_epi32(a1, 32);
	const __m256i a3 = _mm256_shuffle_epi32(a2, 0b11110101);
	const __m256i a4 = _mm256_blendv_epi8(tt, zz, a3);

	alignas(32) uint64_t result[4] = {};
	_mm256_storeu_si256((__m256i*)result, a4);

	result[0] = std::min(result[0], result[1]);
	result[0] = std::min(result[0], result[2]);
	result[0] = std::min(result[0], result[3]);
	return result[0];
}

void unittest_unique() {
	std::mt19937_64 rnd(12345);

	for (int i = 0; i < 10000; ++i) {
		const uint64_t x = rnd();

		const uint64_t y1 = get_unique_naive(x);
		const uint64_t y2 = get_unique_avx2_1(x);
		const uint64_t y3 = get_unique_avx2_2(x);
		assert(y1 == y2);
		assert(y1 == y3);
	}
}

uint64_t bit_reverse_naive(uint64_t b) {
	return horizontal_mirror(vertical_mirror(b));
}

uint64_t bit_reverse_simd(uint64_t b) {

	constexpr auto f = [](const uint8_t x) {
		return uint8_t(((x & 1) << 3) + ((x & 2) << 1) + ((x & 4) >> 1) + ((x & 8) >> 3));
	};

	const __m128i r1 = _mm_set_epi8(f(15), f(14), f(13), f(12), f(11), f(10), f(9), f(8), f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0));

	const __m128i a1 = _mm_set_epi64x((b >> 4) & 0x0F0F'0F0F'0F0F'0F0FULL, b & 0x0F0F'0F0F'0F0F'0F0FULL);
	const __m128i a2 = _mm_shuffle_epi8(r1, a1);
	const __m128i a3 = _mm_shuffle_epi8(a2, _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7));

	alignas(32)uint64_t c[2] = {};
	_mm_storeu_si128((__m128i*)c, a3);
	return (c[0] << 4) + c[1];
}

void unittest_reverse() {
	std::mt19937_64 rnd(12345);

	for (int i = 0; i < 10000; ++i) {
		const uint64_t x = rnd();

		const uint64_t y = bit_reverse_naive(x);

		for (int i = 0; i < 64; ++i) {
			assert(((x & (1ULL << i)) == 0) == ((y & (1ULL << (63 - i))) == 0));
		}

		const uint64_t y1 = bit_reverse_simd(x);
		assert(y == y1);

	}
}

alignas(32) const static uint8_t transpose_5x5_table[32] = { 0,5,10,15,20,1,6,11,16,21,2,7,12,17,22,3,8,13,18,23,4,9,14,19,24,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };

alignas(32) const static uint8_t vertical_mirror_5x5_table[32] = { 20,21,22,23,24,15,16,17,18,19,10,11,12,13,14,5,6,7,8,9,0,1,2,3,4,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };

alignas(32) const static uint8_t horizontal_mirror_5x5_table[32] = { 4,3,2,1,0,9,8,7,6,5,14,13,12,11,10,19,18,17,16,15,24,23,22,21,20,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };

uint64_t code_unique1(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	_mm256_storeu_si256((__m256i*)result1, tt);
	_mm256_storeu_si256((__m256i*)result2, zz);

	for (int i = 0; i < 4; ++i) {
		result1[i] = std::min(result1[i], result2[i]);
	}
	for (int i = 1; i < 4; ++i) {
		result1[0] = std::min(result1[0], result1[i]);
	}
	return result1[0];

	//const bool b00 = result1[0] < result2[0];
	//const uint64_t r0 = b00 ? result1[0] : result2[0];
	//const bool b01 = result1[1] < result2[1];
	//const uint64_t r1 = b01 ? result1[1] : result2[1];
	//const bool b02 = result1[2] < result2[2];
	//const uint64_t r2 = b02 ? result1[2] : result2[2];
	//const bool b03 = result1[3] < result2[3];
	//const uint64_t r3 = b03 ? result1[3] : result2[3];
	//const bool b10 = r0 < r1;
	//const uint64_t r4 = b10 ? r0 : r1;
	//const bool b11 = r2 < r3;
	//const uint64_t r5 = b11 ? r2 : r3;
	//const bool b20 = r4 < r5;
	//return b20 ? r4 : r5;
}

uint64_t code_unique2(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	_mm256_storeu_si256((__m256i*)result1, tt);
	_mm256_storeu_si256((__m256i*)result2, zz);

	//for (int i = 0; i < 4; ++i) {
	//	result1[i] = std::min(result1[i], result2[i]);
	//}
	//for (int i = 1; i < 4; ++i) {
	//	result1[0] = std::min(result1[0], result1[i]);
	//}
	//return result1[0];

	const bool b00 = result1[0] < result2[0];
	const uint64_t r0 = b00 ? result1[0] : result2[0];
	const bool b01 = result1[1] < result2[1];
	const uint64_t r1 = b01 ? result1[1] : result2[1];
	const bool b02 = result1[2] < result2[2];
	const uint64_t r2 = b02 ? result1[2] : result2[2];
	const bool b03 = result1[3] < result2[3];
	const uint64_t r3 = b03 ? result1[3] : result2[3];
	const bool b10 = r0 < r1;
	const uint64_t r4 = b10 ? r0 : r1;
	const bool b11 = r2 < r3;
	const uint64_t r5 = b11 ? r2 : r3;
	const bool b20 = r4 < r5;
	return b20 ? r4 : r5;
}

uint64_t code_unique3(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	const __m256i a1 = _mm256_sub_epi64(zz, tt);
	const __m256i a2 = _mm256_srai_epi32(a1, 32);
	const __m256i a3 = _mm256_shuffle_epi32(a2, 0b11110101);
	const __m256i a4 = _mm256_blendv_epi8(tt, zz, a3);
	alignas(32) uint64_t result[4] = {};
	_mm256_storeu_si256((__m256i*)result, a4);

	result[0] = std::min(result[0], result[1]);
	result[0] = std::min(result[0], result[2]);
	result[0] = std::min(result[0], result[3]);
	return result[0];

	//alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	//_mm256_storeu_si256((__m256i*)result1, tt);
	//_mm256_storeu_si256((__m256i*)result2, zz);

	//for (int i = 0; i < 4; ++i) {
	//	result1[i] = std::min(result1[i], result2[i]);
	//}
	//for (int i = 1; i < 4; ++i) {
	//	result1[0] = std::min(result1[0], result1[i]);
	//}
	//return result1[0];

	//const bool b00 = result1[0] < result2[0];
	//const uint64_t r0 = b00 ? result1[0] : result2[0];
	//const bool b01 = result1[1] < result2[1];
	//const uint64_t r1 = b01 ? result1[1] : result2[1];
	//const bool b02 = result1[2] < result2[2];
	//const uint64_t r2 = b02 ? result1[2] : result2[2];
	//const bool b03 = result1[3] < result2[3];
	//const uint64_t r3 = b03 ? result1[3] : result2[3];
	//const bool b10 = r0 < r1;
	//const uint64_t r4 = b10 ? r0 : r1;
	//const bool b11 = r2 < r3;
	//const uint64_t r5 = b11 ? r2 : r3;
	//const bool b20 = r4 < r5;
	//return b20 ? r4 : r5;
}

uint64_t code_unique4(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	const __m256i a1 = _mm256_sub_epi64(zz, tt);
	const __m256i a2 = _mm256_srai_epi32(a1, 32);
	const __m256i a3 = _mm256_shuffle_epi32(a2, 0b11110101);
	const __m256i a4 = _mm256_blendv_epi8(tt, zz, a3);
	alignas(32) uint64_t result[4] = {};
	_mm256_storeu_si256((__m256i*)result, a4);

	result[0] = std::min(result[0], result[1]);
	result[2] = std::min(result[2], result[3]);
	result[0] = std::min(result[0], result[2]);
	return result[0];

	//alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	//_mm256_storeu_si256((__m256i*)result1, tt);
	//_mm256_storeu_si256((__m256i*)result2, zz);

	//for (int i = 0; i < 4; ++i) {
	//	result1[i] = std::min(result1[i], result2[i]);
	//}
	//for (int i = 1; i < 4; ++i) {
	//	result1[0] = std::min(result1[0], result1[i]);
	//}
	//return result1[0];

	//const bool b00 = result1[0] < result2[0];
	//const uint64_t r0 = b00 ? result1[0] : result2[0];
	//const bool b01 = result1[1] < result2[1];
	//const uint64_t r1 = b01 ? result1[1] : result2[1];
	//const bool b02 = result1[2] < result2[2];
	//const uint64_t r2 = b02 ? result1[2] : result2[2];
	//const bool b03 = result1[3] < result2[3];
	//const uint64_t r3 = b03 ? result1[3] : result2[3];
	//const bool b10 = r0 < r1;
	//const uint64_t r4 = b10 ? r0 : r1;
	//const bool b11 = r2 < r3;
	//const uint64_t r5 = b11 ? r2 : r3;
	//const bool b20 = r4 < r5;
	//return b20 ? r4 : r5;
}

uint64_t code_unique5(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	const __m256i a1 = _mm256_sub_epi64(zz, tt);
	const __m256i a2 = _mm256_srai_epi32(a1, 32);
	const __m256i a3 = _mm256_shuffle_epi32(a2, 0b11110101);
	const __m256i a4 = _mm256_blendv_epi8(tt, zz, a3);

	const __m128i a5 = _mm256_extracti128_si256(a4, 0);
	const __m128i a6 = _mm256_extracti128_si256(a4, 1);

	const __m128i a7 = _mm_sub_epi64(a5, a6);
	const __m128i a8 = _mm_srai_epi32(a7, 32);
	const __m128i a9 = _mm_shuffle_epi32(a8, 0b11110101);
	const __m128i aa = _mm_blendv_epi8(a6, a5, a9);

	alignas(32) uint64_t result[2] = {};
	_mm_storeu_si128((__m128i*)result, aa);

	result[0] = std::min(result[0], result[1]);
	return result[0];

	//alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	//_mm256_storeu_si256((__m256i*)result1, tt);
	//_mm256_storeu_si256((__m256i*)result2, zz);

	//for (int i = 0; i < 4; ++i) {
	//	result1[i] = std::min(result1[i], result2[i]);
	//}
	//for (int i = 1; i < 4; ++i) {
	//	result1[0] = std::min(result1[0], result1[i]);
	//}
	//return result1[0];

	//const bool b00 = result1[0] < result2[0];
	//const uint64_t r0 = b00 ? result1[0] : result2[0];
	//const bool b01 = result1[1] < result2[1];
	//const uint64_t r1 = b01 ? result1[1] : result2[1];
	//const bool b02 = result1[2] < result2[2];
	//const uint64_t r2 = b02 ? result1[2] : result2[2];
	//const bool b03 = result1[3] < result2[3];
	//const uint64_t r3 = b03 ? result1[3] : result2[3];
	//const bool b10 = r0 < r1;
	//const uint64_t r4 = b10 ? r0 : r1;
	//const bool b11 = r2 < r3;
	//const uint64_t r5 = b11 ? r2 : r3;
	//const bool b20 = r4 < r5;
	//return b20 ? r4 : r5;
}

void unittest() {
	std::mt19937 rnd(12345);

	for (int i = 0; i < 10000; ++i) {
		const uint64_t x = ((rnd() % (1ULL << 50)) << 5) + (rnd() % 25);

		const uint64_t y1 = code_unique1(x);
		const uint64_t y2 = code_unique2(x);
		const uint64_t y3 = code_unique3(x);
		const uint64_t y4 = code_unique4(x);
		const uint64_t y5 = code_unique5(x);
		assert(y1 == y2 && y1 == y3 && y1 == y4 && y1 == y5);
	}
}

inline uint64_t xorshift64(uint64_t x) {
	x = x ^ (x << 7);
	return x ^ (x >> 9);
}

#define DEF_BENCH_T(name) \
void bench_t_##name() {\
	std::cout << "Bench throughput:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		const uint64_t x = ((a % (1ULL << 50)) << 5) + (a % 25);\
		result ^= code_unique##name(x);\
		a = xorshift64(a);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\

#define DEF_BENCH_L(name) \
void bench_l_##name() {\
	std::cout << "Bench latency:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		const uint64_t x = ((a % (1ULL << 50)) << 5) + (a % 25);\
		result ^= code_unique##name(x);\
		a = xorshift64(result);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\

#define DEF_BENCH_REV_T(name) \
void bench_rev_t_##name() {\
	std::cout << "Bench rev throughput:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		const uint64_t x = ((a % (1ULL << 50)) << 5) + (a % 25);\
		result ^= bit_reverse_##name(x);\
		a = xorshift64(a);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\

#define DEF_BENCH_REV_L(name) \
void bench_rev_l_##name() {\
	std::cout << "Bench rev latency:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		const uint64_t x = ((a % (1ULL << 50)) << 5) + (a % 25);\
		result ^= bit_reverse_##name(x);\
		a = xorshift64(result);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\

#define DEF_BENCH_UNIQUE_T(name) \
void bench_unique_t_##name() {\
	std::cout << "Bench unique throughput:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		const uint64_t x = ((a % (1ULL << 50)) << 5) + (a % 25);\
		result ^= get_unique_##name(x);\
		a = xorshift64(a);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\

#define DEF_BENCH_UNIQUE_L(name) \
void bench_unique_l_##name() {\
	std::cout << "Bench unique latency:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		const uint64_t x = ((a % (1ULL << 50)) << 5) + (a % 25);\
		result ^= get_unique_##name(x);\
		a = xorshift64(result);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\

DEF_BENCH_T(1)
DEF_BENCH_T(2)
DEF_BENCH_T(3)
DEF_BENCH_T(4)
DEF_BENCH_T(5)

DEF_BENCH_L(1)
DEF_BENCH_L(2)
DEF_BENCH_L(3)
DEF_BENCH_L(4)
DEF_BENCH_L(5)

DEF_BENCH_REV_T(naive)
DEF_BENCH_REV_T(simd)

DEF_BENCH_REV_L(naive)
DEF_BENCH_REV_L(simd)

DEF_BENCH_UNIQUE_T(naive)
DEF_BENCH_UNIQUE_T(avx2_1)
DEF_BENCH_UNIQUE_T(avx2_2)

DEF_BENCH_UNIQUE_L(naive)
DEF_BENCH_UNIQUE_L(avx2_1)
DEF_BENCH_UNIQUE_L(avx2_2)

int main() {

	unittest_unique();

	bench_unique_l_naive();
	bench_unique_l_avx2_1();
	bench_unique_l_avx2_2();
	bench_unique_t_naive();
	bench_unique_t_avx2_1();
	bench_unique_t_avx2_2();

	unittest_reverse();

	bench_rev_l_naive();
	bench_rev_l_simd();
	bench_rev_t_naive();
	bench_rev_t_simd();

	unittest();

	bench_l_1();
	bench_l_2();
	bench_l_3();
	bench_l_4();
	bench_l_5();

	bench_t_1();
	bench_t_2();
	bench_t_3();
	bench_t_4();
	bench_t_5();

	return 0;
}
