
#include <math.h>

#include <random>
#include <vector>
#include <cassert>
#include <string>
#include <regex>
#include <iostream>
#include <fstream>
#include <functional>
#include <filesystem>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

#include <fftw3.h>

#include <gmp.h>
#include <gmpxx.h>

bool is_prime(const int64_t q) {
	if (q <= 1)return false;
	if (q % 2 == 0)return false;
	for (int64_t j = 3; j * j <= q; j += 2) {
		if (q % j == 0)return false;
	}
	return true;
}

int64_t ceiling(int64_t q, int64_t j, int64_t N) {

	//Ceiling(q*j/N)を求めて返す。

	if ((q * j) % N == 0)return q * j / N;
	if (j < 0)return q * j / N;
	return (q * j / N) + 1;

}

void digit_to_balanced(const std::vector<int64_t>&b, const std::vector<int64_t>&x, std::vector<int64_t>&y) {

	//xがdigit representationだとして、それをbalanced representationに変換したものをyに格納する。

	const int64_t N = b.size();

	assert(x.size() == N && y.size() == N);

	for (int64_t j = 0; j < N; ++j) {
		const int64_t digit_bound = (1LL << b[j]);
		assert(0 <= x[j] && x[j] < digit_bound);
	}

	int64_t carry = 0;
	for (int64_t j = 0; j < N; ++j) {

		const int64_t digit_bound = (1LL << b[j]);
		const int64_t balanced_bound = (1LL << b[j]) / 2;

		y[j] = x[j] + carry;
		carry = 0;
		while (balanced_bound <= y[j]) {
			y[j] -= digit_bound;
			++carry;
		}
	}
	for (; carry;) {
		for (int64_t j = 0; j < N && carry; ++j) {

			const int64_t digit_bound = (1LL << b[j]);
			const int64_t balanced_bound = (1LL << b[j]) / 2;

			y[j] += carry;
			carry = 0;
			while (balanced_bound <= y[j]) {
				y[j] -= digit_bound;
				++carry;
			}
		}
	}

	for (int64_t j = 0; j < N; ++j) {
		const int64_t balanced_bound = (1LL << b[j]) / 2;
		assert(-balanced_bound <= y[j] && y[j] < balanced_bound);
	}

	for (int64_t j = 0; j < N; ++j) {
		if (y[j] != -1)return;
	}
	for (int64_t j = 0; j < N; ++j)y[j] = 0;
}

void balanced_to_digit(const std::vector<int64_t>&b, const std::vector<int64_t>&x, std::vector<int64_t>&y) {

	//xがbalanced representationだとして、それをdigit representationに変換したものをyに格納する。

	const int64_t N = b.size();

	assert(x.size() == N && y.size() == N);

	for (int64_t j = 0; j < N; ++j) {
		const int64_t balanced_bound = (1LL << b[j]) / 2;
		assert(-balanced_bound <= x[j] && x[j] < balanced_bound);
	}

	int64_t carry = 0;
	for (int64_t j = 0; j < N; ++j) {

		const int64_t digit_bound = (1LL << b[j]);

		y[j] = x[j] + carry;
		carry = 0;
		while (y[j] < 0) {
			y[j] += digit_bound;
			--carry;
		}
	}
	for (; carry;) {
		for (int64_t j = 0; j < N && carry; ++j) {

			const int64_t digit_bound = (1LL << b[j]);

			y[j] += carry;
			carry = 0;
			while (y[j] < 0) {
				y[j] += digit_bound;
				--carry;
			}
		}
	}

	for (int64_t j = 0; j < N; ++j) {
		const int64_t digit_bound = (1LL << b[j]);
		assert(0 <= y[j] && y[j] < digit_bound);
	}
	for (int64_t j = 0; j < N; ++j) {
		if (y[j] != (1LL << b[j]) - 1)return;
	}
	for (int64_t j = 0; j < N; ++j)y[j] = 0;
}

void compute_borrow_balanced(const std::vector<int64_t>&b, std::vector<int64_t>&x) {

	//xがbalanced representationだとして、繰り上がり・繰り下がりとmod(2^q-1)の処理をする。

	const int64_t N = b.size();
	assert(x.size() == N);

	int64_t carry = 0;

	const auto func_carry = [&](const int64_t j) {
		const int64_t digit_bound = (1LL << b[j]);
		const int64_t balanced_bound = (1LL << b[j]) / 2;
		x[j] += carry;

		const int64_t signmask = x[j] >> 63;
		const int64_t tmp1 = x[j] * signmask;
		const int64_t abs_xj = signmask ? tmp1 : x[j];
		const int64_t signvalue = signmask * 2 + 1;
		carry = (abs_xj >> b[j]) * signvalue;
		x[j] = (abs_xj & (digit_bound - 1)) * signvalue;

		const int64_t f = ((x[j] < -balanced_bound) ? 1 : 0) + ((balanced_bound <= x[j]) ? -1 : 0);
		x[j] += digit_bound * f;
		carry -= f;
	};

	for (int64_t j = 0; j < N; ++j) {
		func_carry(j);
	}
	for (; carry;) {
		for (int64_t j = 0; j < N && carry; ++j) {
			func_carry(j);
		}
	}
	for (int64_t j = 0; j < N; ++j) {
		if (x[j] != -1)return;
	}
	for (int64_t j = 0; j < N; ++j)x[j] = 0;
}

std::string int2str(const std::vector<int64_t>&b, const std::vector<int64_t> &x) {

	//xがdigit representationだとして、それを16進数の文字列に変換したものを返す。

	const int64_t N = b.size();
	
	assert(x.size() == N);

	std::string answer;

	int64_t number = x[0];
	int64_t keta = b[0];

	auto emit1 = [&]() {
		keta -= 4;
		int64_t n = number % 16;
		number /= 16;
		if (n <= 9)return char('0' + n);
		return char('a' + (n - 10));
	};

	for (int j = 1; j < N; ++j) {
		while (keta >= 4) {
			answer += emit1();
		}
		number += x[j] << keta;
		keta += b[j];
	}
	while (keta >= 1) {
		answer += emit1();
	}

	for (; answer.size() >= 2 && answer.back() == '0'; answer.pop_back());

	std::reverse(answer.begin(), answer.end());
	return answer;
}

void square_complex(const int64_t N, fftw_complex*x) {
	const int64_t M = (N / 4) * 4;
	for (int64_t j = 0; j < M; j += 4) {
		const __m256d x1a = _mm256_loadu_pd(&x[j + 0][0]);
		const __m256d x1b = _mm256_loadu_pd(&x[j + 2][0]);

		const __m256d re = _mm256_unpacklo_pd(x1a, x1b);
		const __m256d im = _mm256_unpackhi_pd(x1a, x1b);

		const __m256d r2 = _mm256_mul_pd(im, im);
		const __m256d r3 = _mm256_fmsub_pd(re, re, r2);

		const __m256d i2 = _mm256_mul_pd(re, im);
		const __m256d i3 = _mm256_add_pd(i2, i2);

		const __m256d y1 = _mm256_unpacklo_pd(r3, i3);
		const __m256d y2 = _mm256_unpackhi_pd(r3, i3);

		_mm256_storeu_pd(&x[j + 0][0], y1);
		_mm256_storeu_pd(&x[j + 2][0], y2);
	}
	for (int64_t j = M; j < N; ++j) {
		const double re = x[j][0] * x[j][0] - x[j][1] * x[j][1];
		const double im = 2.0 * x[j][0] * x[j][1];
		x[j][0] = re;
		x[j][1] = im;
	}
}

void round_value_and_count_error(const int64_t N, const double a[], const double in[], int64_t out[], double &error_sum, double &error_max) {

	error_sum = 0.0;
	error_max = 0.0;

	__m256d tmp_error_sum = _mm256_set1_pd(0.0);
	__m256d tmp_error_max = _mm256_set1_pd(0.0);

	const __m256d N4 = _mm256_set1_pd(double(N));
	const __m256d signmask = _mm256_set1_pd(-0.0);

	const int64_t M = (N / 4) * 4;
	for (int64_t j = 0; j < M; j += 4) {

		const __m256d Na = _mm256_mul_pd(N4, _mm256_loadu_pd(&a[j]));
		const __m256d value = _mm256_div_pd(_mm256_loadu_pd(&in[j]), Na);
		const __m256d rounded_value = _mm256_round_pd(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		const __m256d error = _mm256_sub_pd(value, rounded_value);
		const __m256d absolute_error = _mm256_andnot_pd(signmask, error);

		tmp_error_sum = _mm256_add_pd(tmp_error_sum, absolute_error);
		tmp_error_max = _mm256_max_pd(tmp_error_max, absolute_error);

		const __m256d reversed_value = _mm256_permute_pd(rounded_value, 0b0101);

		out[j + 0] = _mm_cvtsd_si64(_mm256_castpd256_pd128(rounded_value));
		out[j + 1] = _mm_cvtsd_si64(_mm256_castpd256_pd128(reversed_value));
		out[j + 2] = _mm_cvtsd_si64(_mm256_extractf128_pd(rounded_value, 0b1));
		out[j + 3] = _mm_cvtsd_si64(_mm256_extractf128_pd(reversed_value, 0b1));
	}
	for (int64_t j = M; j < N; ++j) {

		const double x  = in[j] / (N * a[j]);
		out[j] = std::round(x);
		const double error_value = std::abs(out[j] - x);
		error_sum += error_value;
		error_max = std::max(error_max, error_value);
	}

	alignas(32) double esum[4];
	alignas(32) double emax[4];
	_mm256_storeu_pd(esum, tmp_error_sum);
	_mm256_storeu_pd(emax, tmp_error_max);
	for (int64_t j = 0; j < 4; ++j){
		error_sum += esum[j];
		error_max = std::max(error_max, emax[j]);
	}
}

void LLtest_IBDWT(const int64_t q, const int64_t N, int maxiter, int threads, double fftw_planning_time_limit) {

	assert(q >= N);

	if (!is_prime(q)) {
		std::cout << "warning: q = "<< q <<" is not a prime. Thus this LL-test returns nonsense result.";
	}

	std::cout << "start: prep: LLtest_IBDWT(q = " << q << ", N = " << N << ", maxiter = " << maxiter << ", threads = " << threads << ", planning time limit = " << fftw_planning_time_limit << ")" << std::endl; 

	bool all_zero_flag = true;
	std::vector<int64_t>x_int_digit(N);

	double *r_r2c2r;
	fftw_complex *c_r2c2r;
	fftw_plan p_r2c, p_c2r;

	fftw_plan_with_nthreads(threads);
	fftw_set_timelimit(fftw_planning_time_limit);
	r_r2c2r = (double*)fftw_malloc(sizeof(double) * N);
	c_r2c2r = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((N / 2) + 1));
	p_r2c = fftw_plan_dft_r2c_1d(N, r_r2c2r, c_r2c2r, FFTW_PATIENT);
	p_c2r = fftw_plan_dft_c2r_1d(N, c_r2c2r, r_r2c2r, FFTW_PATIENT);

	std::vector<int64_t>b(N);
	int64_t sum_b = 0;
	for (int64_t j = 0; j < N; ++j) {
		b[j] = ceiling(q, j + 1, N) - ceiling(q, j, N);
		sum_b += b[j];
	}
	assert(sum_b == q);

	//std::vector<double>a(N, 0.0);
	double *a = (double*)fftw_malloc(sizeof(double) * N);
	for (int64_t j = 0; j < N; ++j) {
		const int64_t index = ceiling(q, j, N) * N - (q * j);
		a[j] = std::pow(2.0, double(index) / double(N));
		assert(1.0 <= a[j] && a[j] <= 2.0);
	}

	std::vector<int64_t>x_int_balanced(N);
	x_int_balanced[0] = 4;

	std::cout << "FFTW3 1D real -> Hermitian conjugate -> real, PATIENT: TIME_LIMIT = " << fftw_planning_time_limit << std::endl;

	const auto start = std::chrono::system_clock::now();

	for (uint32_t iteration = 1; iteration <= q - 2; ++iteration) {

		const auto start_iter = std::chrono::system_clock::now();

		for (int j = 0; j < N; ++j)r_r2c2r[j] = x_int_balanced[j] * a[j];

		fftw_execute(p_r2c);

		square_complex(N / 2 + 1, c_r2c2r);

		fftw_execute(p_c2r);

		double error_sum = 0.0, error_max = 0.0;
		round_value_and_count_error(N, a, r_r2c2r, x_int_balanced.data(), error_sum, error_max);
		if (error_max > 0.25) {
			std::cout << "error: DWT results were not integers. error_sum = " << error_sum << ", error_max = " << error_max << std::endl;
			goto END_PHASE;
		}

		x_int_balanced[0] -= 2;

		const auto start_borrow = std::chrono::system_clock::now();

		compute_borrow_balanced(b, x_int_balanced);

		const auto end = std::chrono::system_clock::now();
		const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		const auto elapsed_iter = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_iter).count();
		const auto elapsed_borrow = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_borrow).count();

		std::cout << "iteration " << iteration << " / " << q - 2 << ", error_sum = " << error_sum << ", error_max = " << error_max << ", totaltime = " << elapsed << " ms, raptime = " << elapsed_iter << " ms, borrow = " << elapsed_borrow << " ms" << std::endl;

		if (iteration == maxiter) {
			balanced_to_digit(b, x_int_balanced, x_int_digit);
			const std::string numstr = int2str(b, x_int_digit);
			std::cout << "number =" << std::endl << "0x" << numstr.substr(numstr.size() - 32) << std::endl;
			goto END_PHASE;
		}
	}


	if (is_prime(q)) {
		all_zero_flag = true;
		for (int j = 0; j < N; ++j) {
			if (x_int_balanced[j] != 0) {
				all_zero_flag = false;
				break;
			}
		}
		if (all_zero_flag) {
			std::cout << "2^" << q << "-1 is a prime!" << std::endl;
		}
		else {
			std::cout << "2^" << q << "-1 is not a prime." << std::endl;
		}
	}

	balanced_to_digit(b, x_int_balanced, x_int_digit);
	std::cout << "number =" << std::endl << "0x" << int2str(b, x_int_digit) << std::endl;

END_PHASE:;

	fftw_free(a);
	fftw_free(r_r2c2r);
	fftw_free(c_r2c2r);
	fftw_destroy_plan(p_r2c);
	fftw_destroy_plan(p_c2r);
	fftw_cleanup_threads();
}

std::string int2str(const mpz_t&x) {

	mpz_class number(x);
	std::string answer = number.get_str(16);
	return answer;
}

void LLtest_GMP(const uint32_t q, int maxiter) {

	if (!is_prime(q)) {
		std::cout << "warning: q = "<< q <<" is not a prime. Thus this LL-test returns nonsense result.";
	}

	std::cout << "start: prep: LLtest_GMP(q = " << q << ", maxiter = " << maxiter << ")" << std::endl;

	mpz_t integer_mp;
	mpz_init(integer_mp);
	mpz_set_ui(integer_mp, 1UL);
	mpz_mul_2exp(integer_mp, integer_mp, q);
	mpz_sub_ui(integer_mp, integer_mp, 1UL);

	mpz_t integer_a1;
	mpz_init(integer_a1);
	mpz_set_ui(integer_a1, 4UL);

	mpz_t integer_lo;
	mpz_t integer_hi;
	mpz_init(integer_lo);
	mpz_init(integer_hi);

	std::cout << "start: LLtest_GMP" << std::endl;

	const auto start = std::chrono::system_clock::now();

	for (uint32_t iteration = 1; iteration <= q - 2; ++iteration) {

		const auto start_iter = std::chrono::system_clock::now();

		mpz_mul(integer_a1, integer_a1, integer_a1);

		mpz_add(integer_a1, integer_a1, integer_mp);
		mpz_sub_ui(integer_a1, integer_a1, 2);

		while (true) {
			const int c = mpz_cmp(integer_mp, integer_a1);
			if (c > 0)break;
			if (c == 0) {
				mpz_set_ui(integer_a1, 0UL);
				break;
			}

			mpz_tdiv_r_2exp(integer_lo, integer_a1, q);
			mpz_tdiv_q_2exp(integer_hi, integer_a1, q);

			mpz_add(integer_a1, integer_lo, integer_hi);
		}

		const auto end = std::chrono::system_clock::now();
		const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		const auto elapsed_iter = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_iter).count();

		std::cout << "iteration " << iteration << " / " << q - 2 << ", time = " << elapsed << " ms, raptime = " << elapsed_iter << " ms" << std::endl;

		if (iteration == maxiter) {
			const std::string numstr = int2str(integer_a1);
			std::cout << "number = " << std::endl;
			std::cout << "0x" << numstr.substr(numstr.size() - 32) << std::endl;
			goto END_PHASE;
		}
	}

	if (is_prime(q)) {
		if (mpz_sgn(integer_a1) == 0 || mpz_cmp(integer_a1, integer_mp) == 0) {
			std::cout << "2^" << q << "-1 is a prime!" << std::endl;
		}
		else {
			std::cout << "2^" << q << "-1 is not a prime." << std::endl;
		}
	}

END_PHASE:;

	mpz_clear(integer_mp);
	mpz_clear(integer_a1);
	mpz_clear(integer_lo);
	mpz_clear(integer_hi);
}

int main() {

	fftw_init_threads();

	const int q = 412876283;
	const int N1 = 1024*1024*24;
	const int iter = 40;
	const int threads = 1;
	const double planning_time_limit = 1000.0;

	LLtest_IBDWT(q, N1, iter, threads, planning_time_limit);

	LLtest_GMP(q, iter);

	return 0;
}