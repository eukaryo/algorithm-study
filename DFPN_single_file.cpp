
/*
 * @date 1998 - 2018
 * @author Richard Delorme
 * @author Toshihiko Okuhara
 * https://github.com/abulmo/edax-reversi
 * https://github.com/okuhara/edax-reversi-AVX
 */

 /*
  * @author Hiroki Takizawa
  * @date 2020
  */

// This source code is licensed under the
// GNU General Public License v3.0

#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cassert>
#include<cstdint>
#include<regex>
#include<random>
#include<cstdint>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

namespace EDAX {

typedef struct Board {
	uint64_t player, opponent;
} Board;

typedef struct Move {
	uint64_t flipped;
	int x;
}Move;

inline uint64_t bit_count(const uint64_t b) {
	return _mm_popcnt_u64(b);
}

inline uint64_t first_bit(uint64_t b) {
	unsigned long index;
	_BitScanForward64(&index, b);
	return (uint64_t) index;
}

uint64_t hash_rank[16][256];

void hash_code_init() {
	std::mt19937_64 rnd(12345);

	for (int i = 0; i < 100000; ++i)volatile uint64_t x = rnd();

	for (int i = 0; i < 16; ++i) for (int j = 0; j < 256; ++j) {
		for (;;) {
			hash_rank[i][j] = rnd();
			if (8 <= bit_count(hash_rank[i][j]) && bit_count(hash_rank[i][j]) <= 56)break;
		}
	}
}

alignas(64) const uint64_t X_TO_BIT[] = {
	0x0000000000000001ULL, 0x0000000000000002ULL, 0x0000000000000004ULL, 0x0000000000000008ULL,
	0x0000000000000010ULL, 0x0000000000000020ULL, 0x0000000000000040ULL, 0x0000000000000080ULL,
	0x0000000000000100ULL, 0x0000000000000200ULL, 0x0000000000000400ULL, 0x0000000000000800ULL,
	0x0000000000001000ULL, 0x0000000000002000ULL, 0x0000000000004000ULL, 0x0000000000008000ULL,
	0x0000000000010000ULL, 0x0000000000020000ULL, 0x0000000000040000ULL, 0x0000000000080000ULL,
	0x0000000000100000ULL, 0x0000000000200000ULL, 0x0000000000400000ULL, 0x0000000000800000ULL,
	0x0000000001000000ULL, 0x0000000002000000ULL, 0x0000000004000000ULL, 0x0000000008000000ULL,
	0x0000000010000000ULL, 0x0000000020000000ULL, 0x0000000040000000ULL, 0x0000000080000000ULL,
	0x0000000100000000ULL, 0x0000000200000000ULL, 0x0000000400000000ULL, 0x0000000800000000ULL,
	0x0000001000000000ULL, 0x0000002000000000ULL, 0x0000004000000000ULL, 0x0000008000000000ULL,
	0x0000010000000000ULL, 0x0000020000000000ULL, 0x0000040000000000ULL, 0x0000080000000000ULL,
	0x0000100000000000ULL, 0x0000200000000000ULL, 0x0000400000000000ULL, 0x0000800000000000ULL,
	0x0001000000000000ULL, 0x0002000000000000ULL, 0x0004000000000000ULL, 0x0008000000000000ULL,
	0x0010000000000000ULL, 0x0020000000000000ULL, 0x0040000000000000ULL, 0x0080000000000000ULL,
	0x0100000000000000ULL, 0x0200000000000000ULL, 0x0400000000000000ULL, 0x0800000000000000ULL,
	0x1000000000000000ULL, 0x2000000000000000ULL, 0x4000000000000000ULL, 0x8000000000000000ULL,
	0, 0 // <- hack for passing move & nomove
};

alignas(64) const uint64_t lmask_v4[66][4] = {
	{ 0x00000000000000fe, 0x0101010101010100, 0x8040201008040200, 0x0000000000000000 },
	{ 0x00000000000000fc, 0x0202020202020200, 0x0080402010080400, 0x0000000000000100 },
	{ 0x00000000000000f8, 0x0404040404040400, 0x0000804020100800, 0x0000000000010200 },
	{ 0x00000000000000f0, 0x0808080808080800, 0x0000008040201000, 0x0000000001020400 },
	{ 0x00000000000000e0, 0x1010101010101000, 0x0000000080402000, 0x0000000102040800 },
	{ 0x00000000000000c0, 0x2020202020202000, 0x0000000000804000, 0x0000010204081000 },
	{ 0x0000000000000080, 0x4040404040404000, 0x0000000000008000, 0x0001020408102000 },
	{ 0x0000000000000000, 0x8080808080808000, 0x0000000000000000, 0x0102040810204000 },
	{ 0x000000000000fe00, 0x0101010101010000, 0x4020100804020000, 0x0000000000000000 },
	{ 0x000000000000fc00, 0x0202020202020000, 0x8040201008040000, 0x0000000000010000 },
	{ 0x000000000000f800, 0x0404040404040000, 0x0080402010080000, 0x0000000001020000 },
	{ 0x000000000000f000, 0x0808080808080000, 0x0000804020100000, 0x0000000102040000 },
	{ 0x000000000000e000, 0x1010101010100000, 0x0000008040200000, 0x0000010204080000 },
	{ 0x000000000000c000, 0x2020202020200000, 0x0000000080400000, 0x0001020408100000 },
	{ 0x0000000000008000, 0x4040404040400000, 0x0000000000800000, 0x0102040810200000 },
	{ 0x0000000000000000, 0x8080808080800000, 0x0000000000000000, 0x0204081020400000 },
	{ 0x0000000000fe0000, 0x0101010101000000, 0x2010080402000000, 0x0000000000000000 },
	{ 0x0000000000fc0000, 0x0202020202000000, 0x4020100804000000, 0x0000000001000000 },
	{ 0x0000000000f80000, 0x0404040404000000, 0x8040201008000000, 0x0000000102000000 },
	{ 0x0000000000f00000, 0x0808080808000000, 0x0080402010000000, 0x0000010204000000 },
	{ 0x0000000000e00000, 0x1010101010000000, 0x0000804020000000, 0x0001020408000000 },
	{ 0x0000000000c00000, 0x2020202020000000, 0x0000008040000000, 0x0102040810000000 },
	{ 0x0000000000800000, 0x4040404040000000, 0x0000000080000000, 0x0204081020000000 },
	{ 0x0000000000000000, 0x8080808080000000, 0x0000000000000000, 0x0408102040000000 },
	{ 0x00000000fe000000, 0x0101010100000000, 0x1008040200000000, 0x0000000000000000 },
	{ 0x00000000fc000000, 0x0202020200000000, 0x2010080400000000, 0x0000000100000000 },
	{ 0x00000000f8000000, 0x0404040400000000, 0x4020100800000000, 0x0000010200000000 },
	{ 0x00000000f0000000, 0x0808080800000000, 0x8040201000000000, 0x0001020400000000 },
	{ 0x00000000e0000000, 0x1010101000000000, 0x0080402000000000, 0x0102040800000000 },
	{ 0x00000000c0000000, 0x2020202000000000, 0x0000804000000000, 0x0204081000000000 },
	{ 0x0000000080000000, 0x4040404000000000, 0x0000008000000000, 0x0408102000000000 },
	{ 0x0000000000000000, 0x8080808000000000, 0x0000000000000000, 0x0810204000000000 },
	{ 0x000000fe00000000, 0x0101010000000000, 0x0804020000000000, 0x0000000000000000 },
	{ 0x000000fc00000000, 0x0202020000000000, 0x1008040000000000, 0x0000010000000000 },
	{ 0x000000f800000000, 0x0404040000000000, 0x2010080000000000, 0x0001020000000000 },
	{ 0x000000f000000000, 0x0808080000000000, 0x4020100000000000, 0x0102040000000000 },
	{ 0x000000e000000000, 0x1010100000000000, 0x8040200000000000, 0x0204080000000000 },
	{ 0x000000c000000000, 0x2020200000000000, 0x0080400000000000, 0x0408100000000000 },
	{ 0x0000008000000000, 0x4040400000000000, 0x0000800000000000, 0x0810200000000000 },
	{ 0x0000000000000000, 0x8080800000000000, 0x0000000000000000, 0x1020400000000000 },
	{ 0x0000fe0000000000, 0x0101000000000000, 0x0402000000000000, 0x0000000000000000 },
	{ 0x0000fc0000000000, 0x0202000000000000, 0x0804000000000000, 0x0001000000000000 },
	{ 0x0000f80000000000, 0x0404000000000000, 0x1008000000000000, 0x0102000000000000 },
	{ 0x0000f00000000000, 0x0808000000000000, 0x2010000000000000, 0x0204000000000000 },
	{ 0x0000e00000000000, 0x1010000000000000, 0x4020000000000000, 0x0408000000000000 },
	{ 0x0000c00000000000, 0x2020000000000000, 0x8040000000000000, 0x0810000000000000 },
	{ 0x0000800000000000, 0x4040000000000000, 0x0080000000000000, 0x1020000000000000 },
	{ 0x0000000000000000, 0x8080000000000000, 0x0000000000000000, 0x2040000000000000 },
	{ 0x00fe000000000000, 0x0100000000000000, 0x0200000000000000, 0x0000000000000000 },
	{ 0x00fc000000000000, 0x0200000000000000, 0x0400000000000000, 0x0100000000000000 },
	{ 0x00f8000000000000, 0x0400000000000000, 0x0800000000000000, 0x0200000000000000 },
	{ 0x00f0000000000000, 0x0800000000000000, 0x1000000000000000, 0x0400000000000000 },
	{ 0x00e0000000000000, 0x1000000000000000, 0x2000000000000000, 0x0800000000000000 },
	{ 0x00c0000000000000, 0x2000000000000000, 0x4000000000000000, 0x1000000000000000 },
	{ 0x0080000000000000, 0x4000000000000000, 0x8000000000000000, 0x2000000000000000 },
	{ 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, 0x4000000000000000 },
	{ 0xfe00000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xfc00000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xf800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xf000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xe000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0xc000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
	{ 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },	// pass
	{ 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }
};

uint64_t get_moves(uint64_t player, uint64_t opponent) {
	__m256i	PP, mOO, MM, flip_l, flip_r, pre_l, pre_r, shift2;
	__m128i	M;
	const __m256i shift1897 = _mm256_set_epi64x(7, 9, 8, 1);
	const __m256i mflipH = _mm256_set_epi64x(0x7e7e7e7e7e7e7e7e, 0x7e7e7e7e7e7e7e7e, -1, 0x7e7e7e7e7e7e7e7e);

	PP = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(player));
	mOO = _mm256_and_si256(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(opponent)), mflipH);

	flip_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(PP, shift1897));
	flip_r = _mm256_and_si256(mOO, _mm256_srlv_epi64(PP, shift1897));
	flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(mOO, _mm256_sllv_epi64(flip_l, shift1897)));
	flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(mOO, _mm256_srlv_epi64(flip_r, shift1897)));
	pre_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(mOO, shift1897));
	pre_r = _mm256_srlv_epi64(pre_l, shift1897);
	shift2 = _mm256_add_epi64(shift1897, shift1897);
	flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
	flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
	flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
	flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
	MM = _mm256_sllv_epi64(flip_l, shift1897);
	MM = _mm256_or_si256(MM, _mm256_srlv_epi64(flip_r, shift1897));

	M = _mm_or_si128(_mm256_castsi256_si128(MM), _mm256_extracti128_si256(MM, 1));
	M = _mm_or_si128(M, _mm_unpackhi_epi64(M, M));
	return _mm_cvtsi128_si64(M) & ~(player | opponent);	// mask with empties
}

bool can_move(uint64_t player, uint64_t opponent) {
	return get_moves(player, opponent) != 0;
}

uint64_t flip(uint64_t pos, uint64_t player, uint64_t opponent) {
	__m256i	flip, outflank, ocontig;
	const __m256i shift1897 = _mm256_set_epi64x(7, 9, 8, 1);
	const __m256i shift1897_2 = _mm256_set_epi64x(14, 18, 16, 2);
	const __m256i mask_flip1897 = _mm256_set_epi64x(0x007e7e7e7e7e7e00, 0x007e7e7e7e7e7e00, 0x00ffffffffffff00, 0x7e7e7e7e7e7e7e7e);

	const __m256i PPPP = _mm256_set1_epi64x(player);
	const __m256i OOOO_masked = _mm256_and_si256(_mm256_set1_epi64x(opponent), mask_flip1897);
	const __m256i pre = _mm256_and_si256(OOOO_masked, _mm256_srlv_epi64(OOOO_masked, shift1897));
	const __m256i mask = _mm256_loadu_si256((__m256i*)lmask_v4[pos]);

	ocontig = _mm256_set1_epi64x(X_TO_BIT[pos]);

	ocontig = _mm256_and_si256(OOOO_masked, _mm256_srlv_epi64(ocontig, shift1897));
	ocontig = _mm256_or_si256(ocontig, _mm256_and_si256(OOOO_masked, _mm256_srlv_epi64(ocontig, shift1897)));
	ocontig = _mm256_or_si256(ocontig, _mm256_and_si256(pre, _mm256_srlv_epi64(ocontig, shift1897_2)));
	ocontig = _mm256_or_si256(ocontig, _mm256_and_si256(pre, _mm256_srlv_epi64(ocontig, shift1897_2)));
	outflank = _mm256_and_si256(_mm256_srlv_epi64(ocontig, shift1897), PPPP);
	flip = _mm256_andnot_si256(_mm256_cmpeq_epi64(outflank, _mm256_setzero_si256()), ocontig);

	ocontig = _mm256_andnot_si256(OOOO_masked, mask);
	ocontig = _mm256_and_si256(ocontig, _mm256_sub_epi64(_mm256_setzero_si256(), ocontig));	// LS1B
	outflank = _mm256_and_si256(ocontig, PPPP);
	flip = _mm256_or_si256(flip, _mm256_and_si256(mask, _mm256_add_epi64(outflank, _mm256_cmpeq_epi64(outflank, ocontig))));

	const __m128i flip2 = _mm_or_si128(_mm256_castsi256_si128(flip), _mm256_extracti128_si256(flip, 1));

	return uint64_t(_mm_cvtsi128_si64(flip2) | _mm_extract_epi64(flip2, 1));
}

uint64_t board_get_hash_code(const Board &board) {
	uint64_t h1, h2;
	const uint8_t *p = (const uint8_t*)&board;

	h1 = hash_rank[0][p[0]];
	h2 = hash_rank[1][p[1]];
	h1 ^= hash_rank[2][p[2]];
	h2 ^= hash_rank[3][p[3]];
	h1 ^= hash_rank[4][p[4]];
	h2 ^= hash_rank[5][p[5]];
	h1 ^= hash_rank[6][p[6]];
	h2 ^= hash_rank[7][p[7]];
	h1 ^= hash_rank[8][p[8]];
	h2 ^= hash_rank[9][p[9]];
	h1 ^= hash_rank[10][p[10]];
	h2 ^= hash_rank[11][p[11]];
	h1 ^= hash_rank[12][p[12]];
	h2 ^= hash_rank[13][p[13]];
	h1 ^= hash_rank[14][p[14]];
	h2 ^= hash_rank[15][p[15]];

	return h1 ^ h2;
}

void board_swap_players(Board *board) {
	const uint64_t tmp = board->player;
	board->player = board->opponent;
	board->opponent = tmp;
}

void board_update(Board *board, const Move &move) {
	board->player ^= (move.flipped | (1ULL << move.x));
	board->opponent ^= move.flipped;
	board_swap_players(board);

}

void board_restore(Board *board, const Move &move) {
	board_swap_players(board);
	board->player ^= (move.flipped | (1ULL << move.x));
	board->opponent ^= move.flipped;
}

void board_pass(Board *board) {
	board_swap_players(board);
}

std::string board_print(const uint64_t player, const uint64_t opponent) {
	Board board;
	board.player = player;
	board.opponent = opponent;

	int i, j, square, x;
	const char *color = "?*O-." + 1;
	uint64_t moves = get_moves(board.player, board.opponent);
	std::string answer;

	answer += "  A B C D E F G H\n";
	for (i = 0; i < 8; ++i) {
		answer += char(i + '1');
		answer += " ";
		for (j = 0; j < 8; ++j) {
			x = i * 8 + j;
			square = 2 - ((board.opponent >> x) & 1) - 2 * ((board.player >> x) & 1);
			if (square == 0 && (moves & (1ULL << x))) ++square;
			answer += color[square];
			answer += " ";
		}
		answer += char(i + '1');

		answer += "\n";
	}
	answer += "  A B C D E F G H\n";
	return answer;
}

};

using namespace EDAX;

constexpr int32_t SCORE_MIN = -64;
constexpr int32_t SCORE_MAX = 64;
constexpr int32_t SCORE_INF = 127;
constexpr uint64_t DFPN_INF = 0x0100'0000'0000'0000ULL;
constexpr bool WPNS_MODE = true;
constexpr int DFPN_TABLESIZE = 25;
constexpr int DFPN_HASH_N_WAY = 10;

typedef struct EntryDFPN {
	uint64_t player, opponent;
	uint64_t proof, disproof;
	int32_t lower_bound, upper_bound;
}EntryDFPN;

std::vector<EntryDFPN> TableDFPN;

void InitTableDdpn() {
	TableDFPN.resize(1ULL << DFPN_TABLESIZE);
	for (uint64_t i = 0; i < (1ULL << DFPN_TABLESIZE); ++i) {
		TableDFPN[i].player = 0;
		TableDFPN[i].opponent = 0;
		TableDFPN[i].proof = 1;
		TableDFPN[i].disproof = 1;
		TableDFPN[i].lower_bound = -SCORE_INF;
		TableDFPN[i].upper_bound = SCORE_INF;
	}
}

void ClearTableDfpn() {
	for (uint64_t i = 0; i < (1ULL << DFPN_TABLESIZE); ++i) {
		TableDFPN[i].proof = 1;
		TableDFPN[i].disproof = 1;
	}
}

bool GetTableDfpn(const Board &board, const uint64_t hash_code, const int winning_threshold, uint64_t *proof, uint64_t *disproof) {

	const uint64_t mask = ((1ULL << DFPN_TABLESIZE) - 1);
	const uint64_t start_index = hash_code & mask;
	const uint64_t stride = (hash_code >> 61) + 1;

	for (uint64_t i = 0, now_index = start_index; i < DFPN_HASH_N_WAY; ++i, now_index = (now_index + stride) & mask) {

		if (TableDFPN[now_index].player == board.player && TableDFPN[now_index].opponent == board.opponent) {

			if (winning_threshold < TableDFPN[now_index].lower_bound) {
				*proof = 0;
				*disproof = DFPN_INF;
				return true;
			}
			if (TableDFPN[now_index].upper_bound <= winning_threshold) {
				*proof = DFPN_INF;
				*disproof = 0;
				return true;
			}

			*proof = TableDFPN[now_index].proof;
			*disproof = TableDFPN[now_index].disproof;
			return true;
		}
		else if (TableDFPN[now_index].player == 0 && TableDFPN[now_index].opponent == 0) {
			*proof = 1ULL;
			*disproof = 1ULL;
			return false;
		}
	}
	*proof = 1ULL;
	*disproof = 1ULL;
	return false;
}

inline uint64_t WritableLevel(const uint64_t index) {
	return
		std::max(TableDFPN[index].proof, TableDFPN[index].disproof) +
		bit_count(TableDFPN[index].player | TableDFPN[index].opponent);
}

void SetTableDfpn(const Board &board, const uint64_t hash_code, const int winning_threshold, const uint64_t proof, const uint64_t disproof) {

	const uint64_t mask = ((1ULL << DFPN_TABLESIZE) - 1);
	const uint64_t start_index = hash_code & mask;
	const uint64_t stride = (hash_code >> 61) + 1;
	uint64_t min_writable_level = 0xFFFFFFFFFFFFFFFFULL;
	uint64_t min_index = 0;

	for (uint64_t i = 0, now_index = start_index; i < DFPN_HASH_N_WAY; ++i, now_index = (now_index + stride) & mask) {
		if (TableDFPN[now_index].player == board.player && TableDFPN[now_index].opponent == board.opponent) {
			if (proof == DFPN_INF) {
				TableDFPN[now_index].upper_bound = winning_threshold;
			}
			else if (disproof == DFPN_INF) {
				TableDFPN[now_index].lower_bound = winning_threshold + 1;
			}
			TableDFPN[now_index].proof = proof;
			TableDFPN[now_index].disproof = disproof;
			return;
		}
		else if (TableDFPN[now_index].player == 0 && TableDFPN[now_index].opponent == 0) {
			TableDFPN[now_index].player = board.player;
			TableDFPN[now_index].opponent = board.opponent;
			if (proof == DFPN_INF) {
				TableDFPN[now_index].upper_bound = winning_threshold;
			}
			else if (disproof == DFPN_INF) {
				TableDFPN[now_index].lower_bound = winning_threshold + 1;
			}
			TableDFPN[now_index].proof = proof;
			TableDFPN[now_index].disproof = disproof;
			return;
		}
		const uint64_t level = WritableLevel(now_index);
		if (level < min_writable_level) {
			min_writable_level = level;
			min_index = now_index;
		}
	}
	if (TableDFPN[min_index].proof < proof) {
		TableDFPN[min_index].player = board.player;
		TableDFPN[min_index].opponent = board.opponent;
		if (proof == DFPN_INF) {
			TableDFPN[min_index].lower_bound = -SCORE_INF;
			TableDFPN[min_index].upper_bound = winning_threshold;
		}
		else if (disproof == DFPN_INF) {
			TableDFPN[min_index].lower_bound = winning_threshold + 1;
			TableDFPN[min_index].upper_bound = SCORE_INF;
		}
		else {
			TableDFPN[min_index].lower_bound = -SCORE_INF;
			TableDFPN[min_index].upper_bound = SCORE_INF;
		}
		TableDFPN[min_index].proof = proof;
		TableDFPN[min_index].disproof = disproof;
	}
}

inline uint64_t DfpnAdd(const uint64_t a, const uint64_t b) {
	return std::min(a + b, DFPN_INF);
}

inline uint64_t DfpnSub(const uint64_t a, const uint64_t b) {
	assert(a >= b);
	return (a == DFPN_INF) ? a : (a - b);
}

inline uint64_t EpsilonTrick(const uint64_t a) {
	return std::min(a + 1 + (a >> 2), DFPN_INF);
}

inline int32_t board_solve(const Board &board, const int64_t n_empties) {
	const int32_t n_discs_p = bit_count(board.player);
	const int32_t n_discs_o = 64 - n_empties - n_discs_p;
	int32_t score = n_discs_p - n_discs_o;

	if (score < 0) score -= n_empties;
	else if (score > 0) score += n_empties;

	return score;
}

void DfpnMid(
	Board *board,
	const uint64_t proof_number_threshold,
	const uint64_t disproof_number_threshold,
	const int winning_threshold) {

	//コンピュータ数学シリーズ7 ゲーム計算メカニズム p118 プログラム11.2

	uint64_t proof_number, disproof_number;

	const uint64_t hash_code = board_get_hash_code(*board);
	const uint64_t bb_moves = get_moves(board->player, board->opponent);
	const uint64_t n_empties = bit_count(~(board->player | board->opponent));

	if (bb_moves == 0) { // no moves
		if (can_move(board->opponent, board->player)) { // pass

			board_pass(board);
			const uint64_t next_hash_code = board_get_hash_code(*board);
			DfpnMid(board, disproof_number_threshold, proof_number_threshold, -winning_threshold + 1);
			uint64_t p = 0, d = 0;
			GetTableDfpn(*board, next_hash_code, -winning_threshold + 1, &p, &d);
			board_pass(board);

			SetTableDfpn(*board, hash_code, winning_threshold, d, p);
			return;
		}
		else { // game-over

			const int score = board_solve(*board, n_empties);
			if (winning_threshold <= score) {
				proof_number = 0;
				disproof_number = DFPN_INF;
			}
			else {
				proof_number = DFPN_INF;
				disproof_number = 0;
			}
			SetTableDfpn(*board, hash_code, winning_threshold, proof_number, disproof_number);
			return;
		}
	}

	//合法手を列挙する。
	int position[64] = { 0 }, num_position = 0;
	for (uint64_t b = bb_moves; b; b &= b - 1) {
		position[num_position++] = first_bit(b);
	}

	//各子ノードに反復深化を行う。
	Move m;
	while (true) {

		uint64_t MinDisproofNumberInChildren = DFPN_INF;
		uint64_t SumProofNumberInChildren = 0;
		uint64_t phi_child = DFPN_INF, delta_2 = 0;
		uint64_t delta_child = DFPN_INF;
		int child_move = 0;

		if (WPNS_MODE) {
			uint64_t max_num = 0;
			uint64_t num = 0;
			bool select_child_flag = false;
			for (int i = 0; i < num_position; ++i) {

				m.x = position[i];
				m.flipped = flip(position[i], board->player, board->opponent);
				uint64_t phi = 1, delta = 1;

				board_update(board, m);
				const uint64_t next_hash_code = board_get_hash_code(*board);
				GetTableDfpn(*board, next_hash_code, -winning_threshold + 1, &phi, &delta);
				board_restore(board, m);

				max_num = std::max<uint64_t>(max_num, phi);
				if (delta < MinDisproofNumberInChildren)MinDisproofNumberInChildren = delta;

				if (!select_child_flag) {
					if (delta < delta_child) {
						delta_2 = delta_child;
						phi_child = phi;
						delta_child = delta;
						child_move = position[i];
					}
					else if (delta < delta_2) {
						delta_2 = delta;
					}
					if (phi == DFPN_INF)select_child_flag = true;
				}
			}
			//080307のパワポでは(max(証明数)+未解決の指し手数-1)だが
			//GAME-TREE SEARCH USING PROOF NUMBERS THE FIRST TWENTY YEARSでは(max(証明数)+指し手数-1)になっている。
			//これは後者。
			SumProofNumberInChildren = DfpnAdd(max_num, num_position - 1);
		}
		else {
			bool select_child_flag = false;
			for (int i = 0; i < num_position; ++i) {

				m.x = position[i];
				m.flipped = flip(position[i], board->player, board->opponent);
				uint64_t phi = 1, delta = 1;

				board_update(board, m);
				const uint64_t next_hash_code = board_get_hash_code(*board);
				GetTableDfpn(*board, next_hash_code, -winning_threshold + 1, &phi, &delta);
				board_restore(board, m);

				SumProofNumberInChildren = DfpnAdd(SumProofNumberInChildren, phi);
				if (delta < MinDisproofNumberInChildren)MinDisproofNumberInChildren = delta;

				if (!select_child_flag) {
					if (delta < delta_child) {
						delta_2 = delta_child;
						phi_child = phi;
						delta_child = delta;
						child_move = position[i];
					}
					else if (delta < delta_2) {
						delta_2 = delta;
					}
					if (phi == DFPN_INF)select_child_flag = true;
				}
			}
		}

		if (proof_number_threshold <= MinDisproofNumberInChildren ||
			disproof_number_threshold <= SumProofNumberInChildren) {
			SetTableDfpn(*board, hash_code, winning_threshold, MinDisproofNumberInChildren, SumProofNumberInChildren);
			break;
		}

		m.x = child_move;
		m.flipped = flip(child_move, board->player, board->opponent);

		const uint64_t proof_number_next = DfpnSub(DfpnAdd(disproof_number_threshold, phi_child), SumProofNumberInChildren);
		//const uint64_t disproof_number_next = std::min(proof_number_threshold, DfpnAdd(delta_2, 1));
		const uint64_t disproof_number_next = std::min(proof_number_threshold, EpsilonTrick(delta_2));

		board_update(board, m);
		DfpnMid(board, proof_number_next, disproof_number_next, -winning_threshold + 1);
		board_restore(board, m);
	}
}

int DfpnRoot(Board *board, const int winning_threshold) {

	ClearTableDfpn();

	DfpnMid(board, DFPN_INF - 1, DFPN_INF - 1, winning_threshold);

	uint64_t phi = 0, delta = 0;
	const uint64_t hash_code = board_get_hash_code(*board);
	GetTableDfpn(*board, hash_code, winning_threshold, &phi, &delta);
	if (delta == DFPN_INF)return SCORE_INF;
	if (phi == DFPN_INF)return -SCORE_INF;
	assert(0);
	return 0;
}

int DfpnMtdfRoot(const uint64_t player, const uint64_t opponent) {

	Board board;
	board.player = player;
	board.opponent = opponent;

	InitTableDdpn();

	//真の解は[lb, ub]の中にある。
	int lower_bound = SCORE_MIN, upper_bound = SCORE_MAX;
	while (lower_bound + 1 < upper_bound) {
		const int mid = (lower_bound + upper_bound) / 2;
		const int score = DfpnRoot(&board, mid);
		if (score == SCORE_INF)lower_bound = mid;
		else upper_bound = mid;
	}

	assert(lower_bound + 1 == upper_bound);
	return (lower_bound % 2 == 0) ? lower_bound : upper_bound;
}

int main() {

	hash_code_init();

	std::vector<std::pair<std::string, std::string>>fforum1_19{
		std::make_pair("--XXXXX--OOOXX-O-OOOXXOX-OXOXOXXOXXXOXXX--XOXOXX-XXXOOO--OOOOO--","X"),
		std::make_pair("-XXXXXX---XOOOO--XOXXOOX-OOOOOOOOOOOXXOOOOOXXOOX--XXOO----XXXXX-","X"),
		std::make_pair("----OX----OOXX---OOOXX-XOOXXOOOOOXXOXXOOOXXXOOOOOXXXXOXO--OOOOOX","X"),
		std::make_pair("-XXXXXX-X-XXXOO-XOXXXOOXXXOXOOOX-OXOOXXX--OOOXXX--OOXX----XOXXO-","X"),
		std::make_pair("-OOOOO----OXXO-XXXOXOXX-XXOXOXXOXXOOXOOOXXXXOO-OX-XOOO---XXXXX--","X"),

		std::make_pair("--OXXX--OOOXXX--OOOXOXO-OOXOOOX-OOXXXXXXXOOXXOX--OOOOX---XXXXXX-","X"),
		std::make_pair("--OXXO--XOXXXX--XOOOXXXXXOOXXXXXXOOOOXXX-XXXXXXX--XXOOO----XXOO-","X"),
		std::make_pair("---X-X--X-XXXX--XXXXOXXXXXXOOOOOXXOXXXO-XOXXXXO-XOOXXX--XOOXXO--","O"),
		std::make_pair("--XOXX--O-OOXXXX-OOOXXXX-XOXXXOXXXOXOOOXOXXOXOXX--OXOO----OOOO--","O"),
		std::make_pair("-XXXX-----OXXX--XOXOXOXXOXOXXOXXOXXOXOOOXXXOXOOX--OXXO---OOOOO--","O"),

		std::make_pair("---O-XOX----XXOX---XXOOXO-XXOXOXXXXOOXOX-XOOXXXXXOOOXX-XOOOOOOO-","O"),
		std::make_pair("--O--O--X-OOOOX-XXOOOXOOXXOXOXOOXXOXXOOOXXXXOOOO--OXXX---XXXXX--","O"),
		std::make_pair("--XXXXX--OOOXX---OOOXXXX-OXOXOXXOXXXOXXX--XOXOXX--OXOOO--OOOOO--","X"),
		std::make_pair("--XXXXX---OOOX---XOOXXXX-OOOOOOOOOOXXXOOOOOXXOOX--XXOO----XXXXX-","X"),
		std::make_pair("----O------OOX---OOOXX-XOOOXOOOOOXXOXXOOOXXXOOOOOXXXOOXO--OOOOOX","X"),

		std::make_pair("-XXXXXX-X-XXXOO-XOXXXOOXXOOXXXOX-OOOXXXX--OOXXXX---OOO----XOX-O-","X"),
		std::make_pair("-OOOOO----OXXO-XXXOOOXX-XXOXOXXOXXOOXOOOXXXXOO-OX-XOO----XXXX---","X"),
		std::make_pair("-XXX------OOOX--XOOOOOXXOXOXOOXXOXXOOOOOXXXOXOOX--OXXO---OOOOO--","X"),
		std::make_pair("--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-","X")
	};

	for (int i = 0; i < 19; ++i) {

		std::string start_board = fforum1_19[i].first;
		std::string start_color = fforum1_19[i].second;

		if (!std::regex_match(start_board, std::regex(R"(^(-|X|O){64}$)"))) {
			std::cout << "invalid board 1" << std::endl;
			return 1;
		}
		if (start_color.size() != 1 || (start_color[0] != 'X' && start_color[0] != 'O')) {
			std::cout << "invalid board 2" << std::endl;
			return 1;
		}

		uint64_t player = 0;
		uint64_t opponent = 0;
		for (int i = 0; i < 64; ++i) {
			if (start_board[i] == start_color[0])player += 1ULL << i;
			else if (start_board[i] != '-')opponent += 1ULL << i;
		}
		std::cout << "#" << i + 1 << " (black's turn)" << std::endl;
		std::cout << board_print(player, opponent) << std::flush;

		const int p = DfpnMtdfRoot(player, opponent);
		std::cout <<"score = " <<  (p < 0 ? "-" : "+") << (std::abs(p) < 10 ? "0" : "") << std::abs(p) << " (for black)" << std::flush;
		std::cout << std::endl << std::endl;
	}

	return 0;
}

