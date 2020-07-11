
#include <iostream>
#include <chrono>
#include <cassert>

#include <gmp.h>

void LLtest(const uint32_t p) {

	mpz_t interger_mp;
	mpz_init(interger_mp);
	mpz_set_ui(interger_mp, 1UL);
	mpz_mul_2exp(interger_mp, interger_mp, p);
	mpz_sub_ui(interger_mp, interger_mp, 1UL);

	mpz_t interger_a1;
	mpz_init(interger_a1);
	mpz_set_ui(interger_a1, 4UL);

	mpz_t interger_lo;
	mpz_t interger_hi;
	mpz_init(interger_lo);
	mpz_init(interger_hi);

	for (uint32_t i = 1; i <= p - 2; ++i) {

		const auto start = std::chrono::system_clock::now();

		//↓プロファイリング結果これが97%以上を占めている。
		mpz_mul(interger_a1, interger_a1, interger_a1);

		mpz_add(interger_a1, interger_a1, interger_mp);
		mpz_sub_ui(interger_a1, interger_a1, 2);

		while (true) {
			const int c = mpz_cmp(interger_mp, interger_a1);
			if (c > 0)break;
			if (c == 0) {
				mpz_set_ui(interger_a1, 0UL);
				break;
			}

			mpz_tdiv_r_2exp(interger_lo, interger_a1, p);
			mpz_tdiv_q_2exp(interger_hi, interger_a1, p);

			mpz_add(interger_a1, interger_lo, interger_hi);
		}

		const auto end = std::chrono::system_clock::now();
		const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		std::cout << "iteration " << i << " / " << p - 2 << ", time = " << elapsed << " ms" << std::endl;
	}

	if (mpz_sgn(interger_a1) == 0) {
		std::cout << "2^" << p << "-1 is a prime!" << std::endl;
	}
	else {
		std::cout << "2^" << p << "-1 is not a prime." << std::endl;
	}

	mpz_clear(interger_mp);
	mpz_clear(interger_a1);
	mpz_clear(interger_lo);
	mpz_clear(interger_hi);
}

int main() {

	const uint32_t p = 332192897;

	std::cout << "start: Lucas-Lehmer test, p = " << p << std::endl;

	LLtest(p);

	return 0;
}
