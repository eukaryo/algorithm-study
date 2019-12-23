#include<iostream>
#include<iomanip>
#include<map>
#include<unordered_map>
#include<set>
#include<unordered_set>
#include<vector>
#include<array>
#include<string>
#include<stack>
#include<queue>
#include<algorithm>
#include<iterator>
#include<cassert>
#include<functional>
#include<random>
#include<complex>
#include<bitset>
#include<cstdint>
#include<chrono>
#define int int64_t
#define uint uint64_t
#define REP(i, a, b) for (int64_t i = (int64_t)(a); i < (int64_t)(b); i++)
#define rep(i, a) REP(i, 0, a)
#define EACH(i, a) for (auto i: a)
#define ITR(x, a) for (auto x = a.begin(); x != a.end(); x++)
#define ALL(a) (a.begin()), (a.end())
#define HAS(a, x) (a.find(x) != a.end())
#define Min(x) *min_element(ALL(x))
#define Max(x) *max_element(ALL(x))
#define chmax(a, b) (((a) < (b)) ? (b) : (a))
#define chmin(a, b) (((a) < (b)) ? (a) : (b))
#define intmax (std::numeric_limits<int64_t>::max() / 4)
using namespace std;
//typedef boost::multiprecision::cpp_int bigint;
const double EPS = 1e-9;
const double PI = acos(-1.0);

using namespace std;



//ワーシャルフロイド法
//rep(k, N)rep(i, N)rep(j, N)dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);

//maskの部分集合を列挙
//for(int i = mask; i; i = (i - 1) & mask)

//整数乱数
//mt19937_64 rnd(4649);
//uniform_int_distribution<int> r(0, N - 1);

//vector<vector<modint>>combi(N + 1, vector<modint>(N + 1));
//combi[0][0] = 1;
//REP(i, 1, N + 1) {
//	combi[i][0] = combi[i][i] = 1;
//	REP(j, 1, i) combi[i][j] = combi[i - 1][j - 1] + combi[i - 1][j];
//}

//降順にソート、降順に優先度付きキュー
//sort(ALL(v),greater<int>());
//priority_queue<int, vector<int>, greater<int>>Q;

int counting(const multiset<int>&ms, int L, int R) {
	//msの要素のうち、[L,R)であるようなものの数を数えて返す。
	return distance(ms.lower_bound(L), ms.lower_bound(R));
}

string zeroume_(const int printnum, const int maxnum) {
	string ans;
	rep(i, to_string(maxnum).size() - to_string(printnum).size())ans += "0";
	return ans + to_string(printnum);
}

class modint {
	//MODが素数であることを前提として実装してあるが、その判定はしていない。
	//あまりが出るような除算をしてはいけない。
private:
	static const int MOD = 1000000007;
public:
	modint() {
		//assert(is_prime(MOD));
		this->number = 0;
	}
	modint(const int src) {
		//assert(is_prime(MOD));
		this->number = opposit(src);
	}
	modint(const modint &src) {
		this->number = src.number;
	}

	modint& operator += (const modint& obj) {
		this->number = san2(this->number + obj.number);
		return *this;
	}
	modint& operator -= (const modint& obj) {
		this->number = san2(this->number - obj.number + MOD);
		return *this;
	}
	modint& operator *= (const modint& obj) {
		this->number = (this->number * obj.number) % MOD;
		return *this;
	}
	modint& operator /= (const modint& obj) {
		this->number = (this->number * inverse(obj.number)) % MOD;
		return *this;
	}
	modint& operator += (const int n) {
		this->number = san2(this->number + opposit(n));
		return *this;
	}
	modint& operator -= (const int n) {
		this->number = san2(this->number - opposit(n) + MOD);
		return *this;
	}
	modint& operator *= (const int n) {
		this->number = (this->number * opposit(n)) % MOD;
		return *this;
	}
	modint& operator /= (const int n) {
		this->number = (this->number * inverse(n)) % MOD;
		return *this;
	}

	modint operator + (const modint obj) { modint re(*this); return re += obj; }
	modint operator - (const modint obj) { modint re(*this); return re -= obj; }
	modint operator * (const modint obj) { modint re(*this); return re *= obj; }
	modint operator / (const modint obj) { modint re(*this); return re /= obj; }
	modint operator + (const int n) { modint re(*this); return re += n; }
	modint operator - (const int n) { modint re(*this); return re -= n; }
	modint operator * (const int n) { modint re(*this); return re *= n; }
	modint operator / (const int n) { modint re(*this); return re /= n; }

	modint operator = (const int n) {
		this->number = opposit(n);
		return *this;
	}
	int get() {
		return number;
	}

private:
	int number;

	int opposit(int n) {
		if (n < 0)n = MOD - ((-n) % MOD);
		return n % MOD;
	}
	int inverse(int n) {
		n = opposit(n);
		int result = 1;
		for (int i = MOD - 2; i; i /= 2) {
			if (i % 2)result = (result * n) % MOD;
			n = (n * n) % MOD;
		}
		return result;
	}
	inline int san2(const int n) {
		return MOD <= n ? n - MOD : n;
	}
	bool is_prime(int n) {
		if (n <= 1)return false;
		if (n == 2)return true;
		if (n % 2 == 0) return false;
		const int upperbound = int(sqrt(n));
		for (int i = 3; i <= upperbound; i += 2) {
			if (n % i == 0) return false;
		}
		return true;
	}
};
modint power_(modint n, int p) {
	modint result = 1;
	for (; p; p /= 2) {
		if (p % 2)result *= n;
		n *= n;
	}
	return result;
}
modint power_(int n, int p) {
	modint mn = n;
	return power_(mn, p);
}
struct combi {
public:
	vector<modint>facto;
	combi(const int N) :facto(N) {
		facto[0] = 1;
		REP(i, 1, N)facto[i] = facto[i - 1] * i;
	}
	modint get_combi(const int N, const int C) {
		//combination(N+C,C)を求めて返す。
		if (int(facto.size()) <= N + C) {
			int a = facto.size();
			facto.resize(N + C + 1);
			REP(i, a, N + C + 1)facto[i] = facto[i - 1] * i;
		}
		return facto[N + C] / (facto[N] * facto[C]);
	}
};
class BIT_add_sum {
private:

	vector<int>v;
	int N;

public:
	BIT_add_sum(int n) :v(n, 0) { N = n; }

	//v[index]+=numberとする。
	void update(const int index, const int number) {
		for (int x = index; x < N; x |= x + 1) v[x] += number;
	}

	//[0,R)の総和を返す。
	int get(const int R) {
		int result = 0;
		for (int x = R - 1; 0 <= x; x = (x & (x + 1)) - 1) result += v[x];
		return result;
	}

	//[L,R)の総和を返す。
	int get(const int L, const int R) {
		return get(R) - get(L);
	}
};
class BIT_max_max {
private:

	vector<int>v;
	int N;

public:
	BIT_max_max(int n) :v(n, 0) { N = n; }

	//v[index]=max(v[index],number)とする。
	void update(const int index, const int number) {
		for (int x = index; x < N; x |= x + 1) v[x] = max(v[x], number);
	}

	int get(const int index) {//[0,index)の最大値を返す。
		int result = v[index - 1];
		for (int x = index - 1; 0 <= x; x = (x & (x + 1)) - 1) result = max(result, v[x]);
		return result;
	}
};

class segtree_add_sum_lazy {
private:
	typedef int T;
	int SIZE;
	vector<T>all;
	vector<T>part;

	void update_inner(const int queryL, const int queryR, const int index, const int segL, const int segR, const T number) {
		if (queryL <= segL && segR <= queryR)all[index] += number;
		else if (queryL < segR && segL < queryR) {
			part[index] += number * (min(queryR, segR) - max(queryL, segL));
			update_inner(queryL, queryR, index * 2, segL, (segL + segR) / 2, number);
			update_inner(queryL, queryR, index * 2 + 1, (segL + segR) / 2, segR, number);
		}
	}

	T getsum_inner(const int queryL, const int queryR, const int index, const int segL, const int segR) {
		if (queryR <= segL || segR <= queryL)return 0;
		if (queryL <= segL && segR <= queryR)return all[index] * (segR - segL) + part[index];
		T result = all[index] * (min(queryR, segR) - max(queryL, segL));
		result += getsum_inner(queryL, queryR, index * 2, segL, (segL + segR) / 2);
		result += getsum_inner(queryL, queryR, index * 2 + 1, (segL + segR) / 2, segR);
		return result;
	}

	//x以上であるような2のべき乗数のうち最小のものを返す。
	int roundup_pow2(int x) {
		x--;
		rep(i, 6)x |= x >> (1LL << i);
		return x + 1;
	}

public:

	segtree_add_sum_lazy(const int N) {
		SIZE = roundup_pow2(N);
		all.clear();
		all.resize(SIZE * 2, 0);
		part.clear();
		part.resize(SIZE * 2, 0);
	}

	//配列要素の[index]にnumberを加算する。
	void update(const int index, const T number) {
		update(index, index + 1, number);
	}

	//配列要素の[L,R)全てにnumberを加算する。
	void update(const int L, const int R, const T number) {
		update_inner(L, R, 1, 0, SIZE, number);
	}

	//[L,R)の総和を返す。
	T getsum(const int L, const int R) {
		return getsum_inner(L, R, 1, 0, SIZE);
	}
};
class segtree_sum_pointupdate {
private:
	typedef int T;
public:
	segtree_sum_pointupdate(const int N) {
		SIZE = roundup_pow2(N);
		v.clear();
		v.resize(SIZE * 2, 0);
	}

	//v[index]=numberとする。
	void update(const int index, const T number) {
		v[index + SIZE] = number;
		for (int i = (index + SIZE) / 2; i; i = i / 2)v[i] = v[i * 2] + v[i * 2 + 1];
	}

	//[index]の値を返す。
	T getnum(const int index) {
		return v[index + SIZE];
	}

	//[L,R)の総和を返す。
	T getsum(const int L, const int R) {
		return getsum_inner(L, R, 1, 0, SIZE);
	}
private:
	T getsum_inner(const int queryL, const int queryR, const int index, const int segL, const int segR) {
		if (queryR <= segL || segR <= queryL)return 0;
		if (queryL <= segL && segR <= queryR)return v[index];
		return getsum_inner(queryL, queryR, index * 2, segL, (segL + segR) / 2) +
			getsum_inner(queryL, queryR, index * 2 + 1, (segL + segR) / 2, segR);
	}

	//x以上であるような2のべき乗数のうち最小のものを返す。
	int roundup_pow2(int x) {
		x--;
		rep(i, 6)x |= x >> (1LL << i);
		return x + 1;
	}

	int SIZE;
	vector<T>v;
};
class segtree_max_pointupdate {

private:

	typedef int T;
	int SIZE;
	vector<T>v;

	T getmax_inner(const int queryL, const int queryR, const int index, const int segL, const int segR) {
		if (queryR <= segL || segR <= queryL)return numeric_limits<T>::min();
		if (queryL <= segL && segR <= queryR)return v[index];
		return max(getmax_inner(queryL, queryR, index * 2, segL, (segL + segR) / 2),
			getmax_inner(queryL, queryR, index * 2 + 1, (segL + segR) / 2, segR));
	}

	int roundup_pow2(int x) {
		//x以上であるような2のべき乗数のうち最小のものを返す。
		x--;
		rep(i, 6)x |= x >> (1LL << i);
		return x + 1;
	}

public:
	segtree_max_pointupdate(const int N) {
		SIZE = roundup_pow2(N);
		v = vector<int>(SIZE * 2, numeric_limits<T>::min());
	}

	//v[index]=numberとする。
	void update(const int index, const T number) {
		v[index + SIZE] = number;
		for (int i = (index + SIZE) / 2; i; i = i / 2)v[i] = max(v[i * 2], v[i * 2 + 1]);
	}

	//[index]の値を返す。
	T getnum(const int index) {
		return v[index + SIZE];
	}

	//[L,R)の最大値を返す。
	T getmax(const int L, const int R) {
		return getmax_inner(L, R, 1, 0, SIZE);
	}
};

class segtree_min_pointupdate {

private:

	typedef int T;
	int SIZE;
	vector<T>v;

	T getmin_inner(const int queryL, const int queryR, const int index, const int segL, const int segR) {
		if (queryR <= segL || segR <= queryL)return numeric_limits<T>::max();
		if (queryL <= segL && segR <= queryR)return v[index];
		return min(getmin_inner(queryL, queryR, index * 2, segL, (segL + segR) / 2),
			getmin_inner(queryL, queryR, index * 2 + 1, (segL + segR) / 2, segR));
	}

	//x以上であるような2のべき乗数のうち最小のものを返す。
	int roundup_pow2(int x) {
		x--;
		rep(i, 6)x |= x >> (1LL << i);
		return x + 1;
	}

public:
	segtree_min_pointupdate(const int N) {
		SIZE = roundup_pow2(N);
		v.clear();
		v.resize(SIZE * 2, numeric_limits<T>::max());
	}

	//v[index]=numberとする。
	void update(const int index, const T number) {
		v[index + SIZE] = number;
		for (int i = (index + SIZE) / 2; i; i = i / 2)v[i] = min(v[i * 2], v[i * 2 + 1]);
	}

	//[index]の値を返す。
	T getnum(const int index) {
		return v[index + SIZE];
	}

	//[L,R)の最小値を返す。
	T getmin(const int L, const int R) {
		return getmin_inner(L, R, 1, 0, SIZE);
	}
};

//Sparse Tableとは、構築O(NlogN)、一点更新O(N)、RMQがO(1)のデータ構造。
//一旦構築されるとほぼ更新されないデータで大量のRMQを高速処理するのに使える。
template<typename T>class SparseTableMin {
private:

	int64_t rank;
	std::vector<std::vector<T>>v;

public:

	SparseTableMin(const std::vector<T>& input) {
		rank = std::max(int64_t(1), log2_ceiling(input.size()));
		v = std::vector<std::vector<T>>(rank,
			std::vector<T>(input.size(), std::numeric_limits<T>::max()));
		for (int64_t i = 0; i < input.size(); ++i)v[0][i] = input[i];
		for (int64_t x = 1; x < rank; ++x) {
			for (int64_t i = 0; i < input.size(); ++i) {
				const int64_t pos = i + (1 << (x - 1));
				if (pos >= input.size())v[x][i] = v[x - 1][i];
				else v[x][i] = std::min(v[x - 1][i], v[x - 1][pos]);
			}
		}
	}

	SparseTableMin(const int64_t N) {
		SparseTableMin(std::vector<T>(N, std::numeric_limits<T>::max()));
	}
	SparseTableMin() {
		SparseTableMin(1);
	}

	//v[index]=numberとする。O(N)
	void update(const int64_t index, const T number) {
		v[0][index] = number;
		for (int64_t x = 1; x < rank; ++x) {
			for (int64_t i = std::max(int64_t(0), index - (1 << x) + 1);
				i <= index; ++i) {
				const int64_t pos = i + (1 << (x - 1));
				if (pos >= v[0].size())v[x][i] = v[x - 1][i];
				else v[x][i] = std::min(v[x - 1][i], v[x - 1][pos]);
			}
		}
	}

	T getnum(const int64_t index) {
		return v[0][index];
	}

	//[L,R)の最小値を返す。log2_ceilingがO(1)だとみなすとO(1)。厳密にはlog2_ceilingはlog(N)
	T getmin(const int64_t L, const int64_t R) {
		if (L + 1 == R)return v[0][L];
		if (R <= L)return std::numeric_limits<T>::max();
		const int64_t query_length = R - L;
		const int64_t index_rank = log2_ceiling(query_length) - 1;
		return std::min(v[index_rank][L], v[index_rank][R - (1 << index_rank)]);
	}

	static uint64_t popcount(uint64_t x) {
		x = (x & 0x5555555555555555ULL) + ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
		x = (x & 0x3333333333333333ULL) + ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
		x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
		x *= 0x0101010101010101ULL;
		return x >> 56;
	}

	//x以上の2べき乗数のうち最小の数が2^kとしてkを返す。
	static int64_t log2_ceiling(uint64_t x) {

		if (x <= 0)return 0;
		x--;
		//msbの位置を求めるために、ビットを下位に伝播させてpopcntする。
		for (int i = 0; i < 6; ++i)x |= x >> (1ULL << i);
		return popcount(x);
	}
};

//Alstrup の Sparse Tableとは、構築がO(N)に改善されたSparse Tableである。
template<typename T>class AlstrupSparceTableMin {
private:

	int64_t block_size;
	SparseTableMin<T> macro_array;
	std::vector<T>v;
	std::vector<std::vector<uint64_t>>blocks;

	static uint64_t popcount(uint64_t x) {
		x = (x & 0x5555555555555555ULL) + ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
		x = (x & 0x3333333333333333ULL) + ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
		x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
		x *= 0x0101010101010101ULL;
		return x >> 56;
	}

	//x以上の2べき乗数のうち最小の数が2^kとしてkを返す。
	static int64_t log2_ceiling(uint64_t x) {

		if (x <= 0)return 0;
		x--;
		//msbの位置を求めるために、ビットを下位に伝播させてpopcntする。
		for (int i = 0; i < 6; ++i)x |= x >> (1ULL << i);
		return popcount(x);
	}


	T getmin_micro(const int64_t index, const int64_t L, const int64_t R) {
		const uint64_t w = blocks[index][R] & ~((uint64_t(1) << L) - uint64_t(1));
		const int64_t lsb =
			log2_ceiling(w & uint64_t(-int64_t(w)));
		return v[index * block_size + ((w == 0) ? R : lsb)];
	}

public:

	AlstrupSparceTableMin(const std::vector<T>& input) {
		v = input;
		block_size = std::max(int64_t(1),
			log2_ceiling(input.size()) / 2);
		assert(block_size <= 63);
		const int64_t block_num = (input.size() + block_size - 1) / block_size;
		{
			std::vector<int64_t>M(block_num, std::numeric_limits<T>::max());
			for (int64_t i = 0; i < input.size(); ++i) {
				const int64_t index = i / block_size;
				M[index] = std::min(M[index], int64_t(input[i]));
			}
			macro_array = SparseTableMin<T>(M);
		}
		blocks = std::vector<std::vector<uint64_t>>(block_num,
			std::vector<uint64_t>(block_size, 0));
		for (int64_t x = 0; x < blocks.size(); ++x) {
			std::vector<T>B(block_size, std::numeric_limits<T>::max());
			const int64_t offset = x * block_size;
			for (int64_t i = 0;
				offset + i < input.size() && i < block_size; ++i) {
				B[i] = input[offset + i];
			}
			std::vector<int64_t>g(block_size, -1);
			std::stack<int64_t>g_stack;
			for (int64_t i = 0; i < block_size; ++i) {
				while (!g_stack.empty() && B[i] <= B[g_stack.top()]) {
					g_stack.pop();
				}
				g[i] = g_stack.empty() ? -1 : g_stack.top();
				g_stack.push(i);
			}
			for (int64_t i = 1; i < block_size; ++i) {
				blocks[x][i] = g[i] == -1 ? 0 :
					(blocks[x][g[i]] |
					(uint64_t(1) << (uint64_t(g[i]))));
			}
		}
	}

	AlstrupSparceTableMin(const int64_t N) {
		AlstrupSparceTableMin(std::vector<T>(N, std::numeric_limits<T>::max()));
	}

	AlstrupSparceTableMin() {
		AlstrupSparceTableMin(1);
	}

	//v[index]=numberとする。O(N)
	void update(const int64_t index, const T number) {

		v[index] = number;

		const int64_t block_index = index / block_size;
		int64_t macro_min = std::numeric_limits<T>::max();
		const int64_t offset = block_index * block_size;
		for (int64_t i = 0; i < block_size && offset + i < v.size(); ++i) {
			macro_min = std::min(macro_min, v[offset + i]);
		}
		macro_array.update(block_index, macro_min);

		std::vector<T>B(block_size, std::numeric_limits<T>::max());
		for (int64_t i = 0; offset + i < v.size() && i < block_size; ++i) {
			B[i] = v[offset + i];
		}
		std::vector<int64_t>g(block_size, -1);
		std::stack<int64_t>g_stack;
		for (int64_t i = 0; i < block_size; ++i) {
			while (!g_stack.empty() && B[i] <= B[g_stack.top()])g_stack.pop();
			g[i] = g_stack.empty() ? -1 : g_stack.top();
			g_stack.push(i);
		}

		for (int64_t i = 1; i < block_size; ++i) {
			blocks[block_index][i] = g[i] == -1 ? 0 :
				(blocks[block_index][g[i]] |
				(uint64_t(1) << (uint64_t(g[i]))));
		}
	}

	T getnum(const int64_t index) {
		return v[index];
	}

	//[L,R)の最小値を返す。log2_ceilingがO(1)だとみなすとO(1)。厳密にはlog2_ceilingはlog(N)
	int getmin(const int64_t L, const int64_t R) {
		if (R <= L)return std::numeric_limits<T>::max();
		const int64_t L_index = L / block_size;
		const int64_t R_index = (R - 1) / block_size;
		if (L_index == R_index) {
			return getmin_micro(L_index, L % block_size, (R - 1) % block_size);
		}
		T answer = macro_array.getmin(L_index + 1, R_index);
		answer = std::min(
			answer, getmin_micro(L_index, L % block_size, block_size - 1));
		answer = std::min(
			answer, getmin_micro(R_index, 0, (R - 1) % block_size));
		return answer;
	}
};


class union_find {

private:
	vector<int> v;
	vector<int> size;
	int tree_num;

	int find(int a) {
		if (v[a] == a)return a;
		return v[a] = find(v[a]);
	}

public:

	union_find(const int N) :v(N), size(N) {
		tree_num = N;
		rep(i, N)v[i] = i;
		rep(i, N)size[i] = 1;
	}

	void unite(int a, int b) {
		a = find(a);
		b = find(b);
		if (a == b)return;

		if (size[a] < size[b])swap(a, b);
		v[b] = a;
		size[a] += size[b];
		tree_num--;
	}

	bool same(int a, int b) { return find(a) == find(b); }

	int get_size(int a) {
		return size[find(a)];
	}
	int get_tree_num() { return tree_num; }

};

uint64_t popcount(uint64_t x) {
	x = (x & 0x5555555555555555ULL) + ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
	x = (x & 0x3333333333333333ULL) + ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
	x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
	x *= 0x0101010101010101ULL;
	return x >> 56;
}

uint64_t log2_ceiling(uint64_t x) {

	if (x <= 0)return 0;
	x--;
	//msbの位置を求めるために、ビットを下位に伝播させてpopcntする。
	for (int i = 0; i < 6; ++i)x |= x >> (1ULL << i);
	return popcount(x);
}

constexpr uint64_t popcount_constexpr(const uint64_t x, const uint64_t d = 3) {
	return
		d == 3 ? popcount_constexpr((x & 0x5555555555555555ULL) + ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1), 2) :
		d == 2 ? popcount_constexpr((x & 0x3333333333333333ULL) + ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2), 1) :
		d == 1 ? popcount_constexpr((x & 0x0F0F0F0F0F0F0F0FULL) + ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4), 0) :
		((x * 0x0101010101010101ULL) >> 56);
}

constexpr uint64_t log2_ceiling_constexpr(const uint64_t x, const uint64_t i = -1) {

	return (i == -1 && x <= 0) ? 0 :
		i == -1 ? log2_ceiling_constexpr(x - 1, 0) :
		i == 0 ? log2_ceiling_constexpr(x | (x >> 1), 1) :
		i == 1 ? log2_ceiling_constexpr(x | (x >> 2), 2) :
		i == 2 ? log2_ceiling_constexpr(x | (x >> 4), 3) :
		i == 3 ? log2_ceiling_constexpr(x | (x >> 8), 4) :
		i == 4 ? log2_ceiling_constexpr(x | (x >> 16), 5) :
		i == 5 ? log2_ceiling_constexpr(x | (x >> 32), 6) :
		popcount_constexpr(x);
}

constexpr uint64_t median_constexpr(const uint64_t a, const uint64_t b, const uint64_t c) {
	return
		(a <= b && b <= c) ? b :
		(a <= c && c <= b) ? c :
		(b <= a && a <= c) ? a :
		(b <= c && c <= a) ? c :
		(c <= a && a <= b) ? a :
		(c <= b && b <= a) ? b : a;
}


class tree_LCP {
private:

	//log2(木の深さの最大値)<MAX_LOG_Dであることを仮定する。
	static const int MAX_LOG_D = 20;

	//ノード0を根(depth[0]==0)とする。
	vector<int>depth;

	//parent[i][j]:=ノードiから2^j世代親のノードの座標。
	//2^j世代まで遡れないなら0とする。
	vector<array<int, MAX_LOG_D>>parent;

	int ntz(int x) {
		x &= -x;
		x = x | (x >> 1);
		x = x | (x >> 2);
		x = x | (x >> 4);
		x = x | (x >> 8);
		x = x | (x >> 16);
		x = ((x & 0xAAAAAAAA) >> 1) + (x & 0x55555555);
		x = ((x & 0xCCCCCCCC) >> 2) + (x & 0x33333333);
		x = ((x & 0xF0F0F0F0) >> 4) + (x & 0x0F0F0F0F);
		x = ((x & 0xFF00FF00) >> 8) + (x & 0x00FF00FF);
		x = ((x & 0xFFFF0000) >> 16) + (x & 0x0000FFFF);
		return x - 1;
	}
public:
	//引数tは、t[i]:=ノードiとつながっているノードの集合 とする。
	tree_LCP(const vector<vector<int>>&t) :
		depth(t.size(), -1), parent(t.size()) {
		rep(i, parent.size())rep(j, MAX_LOG_D)parent[i][j] = 0;
		stack<array<int, 2>>dfs;
		dfs.push({ 0,0 });
		while (!dfs.empty()) {
			auto x = dfs.top(); dfs.pop();
			const int index = x[0];
			const int d = x[1];
			depth[index] = d;
			for (auto a : t[index])if (a != parent[index][0]) {
				dfs.push({ a,d + 1 });
				parent[a][0] = index;
			}
		}
		REP(i, 1, MAX_LOG_D)rep(j, parent.size()) {
			parent[j][i] = parent[parent[j][i - 1]][i - 1];
		}
	}
	int get_depth(const int a) {
		return depth[a];
	}
	int get_LCP_depth(int a, int b) {
		if (depth[a] < depth[b])swap(a, b);
		for (int i = depth[a] - depth[b]; i; i &= i - 1) {
			a = parent[a][ntz(i)];
		}
		for (int i = MAX_LOG_D - 1; 0 <= i; i--) {
			if (parent[a][i] != parent[b][i]) {
				a = parent[a][i];
				b = parent[b][i];
			}
		}
		return depth[parent[a][0]];
	}
	int get_dist(const int a, const int b) {
		return depth[a] + depth[b] - 2 * get_LCP_depth(a, b);
	}
};

//最小コストフロー問題を解く。
class MinCostFlow {
private:
	struct edge {
		int to, cap, cost, rev; // rev is the position of reverse edge in graph[to]
	};
	typedef std::pair<int, int> P;
	int inf;
	int v; // the number of vertices
	std::vector<std::vector<edge> > graph;
	std::vector<int> h; // potential
	std::vector<int> dist; // minimum distance
	std::vector<int> prevv, preve; // previous vertex and edge
public:
	/* Initializes this solver. v is the number of vertices. */
	MinCostFlow(int v, int infinite = 99999999) :
		v(v), graph(v), h(v), dist(v), prevv(v), preve(v), inf(infinite) {}

	/* Initializes this solver with a existing instance. Only graph is copied. */
	MinCostFlow(const MinCostFlow &ano) : v(ano.v), graph(), h(ano.v), dist(ano.v), prevv(ano.v), preve(ano.v) {
		for (int i = 0; i < ano.v; ++i) {
			std::vector<edge> tt;
			for (int j = 0; j < int(ano.graph[i].size()); ++j) {
				tt.push_back(ano.graph[i][j]);
			}
			graph.push_back(tt);
		}
	}
	/* Adds an edge. */
	void add_edge(int from, int to, int cap, int cost) {
		graph[from].push_back({ to, cap, cost, int(graph[to].size()) });
		graph[to].push_back({ from, 0, -cost, int(graph[from].size() - 1) });
	}
	/* Calcucates the minimum cost flow whose source is s, sink is t, and flow is f. */
	int min_cost_flow(int s, int t, int f) {
		int res = 0;
		std::fill(h.begin(), h.end(), 0);
		while (f > 0) {
			std::priority_queue<P, std::vector<P>, std::greater<P> > que;
			std::fill(dist.begin(), dist.end(), inf);
			dist[s] = 0;
			que.push(P(0, s));
			while (!que.empty()) {
				P p(que.top()); que.pop();
				int v = p.second;
				if (dist[v] < p.first) {
					continue;
				}
				for (int i = 0; i < int(graph[v].size()); ++i) {
					edge &e = graph[v][i];
					if (e.cap > 0 && dist[e.to] > dist[v] + e.cost + h[v] - h[e.to]) {
						dist[e.to] = dist[v] + e.cost + h[v] - h[e.to];
						prevv[e.to] = v;
						preve[e.to] = i;
						que.push(P(dist[e.to], e.to));
					}
				}
			}
			if (dist[t] == inf) {
				return -1; // Cannot add flow anymore
			}
			for (int i = 0; i < v; ++i) {
				h[i] += dist[i];
			}
			// Add flow fully
			int d = f;
			for (int i = t; i != s; i = prevv[i]) {
				d = std::min(d, graph[prevv[i]][preve[i]].cap);
			}
			f -= d;
			res += d * h[t];
			for (int i = t; i != s; i = prevv[i]) {
				edge &e = graph[prevv[i]][preve[i]];
				e.cap -= d;
				graph[i][e.rev].cap += d;
			}
		} // while (f > 0)
		return res;
	}
};

//最大フロー問題をDinicで解く
class MaxFlow {
private:

	struct Edge {
		int to;
		int cap;
		int rev;
		Edge(int to, int cap, int rev) : to(to), cap(cap), rev(rev) {}
	};

	int INF;
	int size;
	vector<int>level;
	vector<int>iter;
	vector<int>que;
	vector<vector<Edge>>graph;


	void bfs(const int from, const int to) {
		int qs = 0, qt = 0;
		for (int i = 0; i < size; i++) level[i] = -1;
		level[from] = 0;
		que[qt++] = from;
		while (qs < qt && level[to] == -1) {
			int now = que[qs++];
			for (int i = 0; i < graph[now].size(); i++) {
				int next = graph[now][i].to;
				if (graph[now][i].cap > 0 && level[next] == -1) {
					level[next] = level[now] + 1;
					que[qt++] = next;
				}
			}
		}
	}

	int dfs(const int from, const int to, const int cap) {
		int flow = 0;
		if (from == to || cap <= 0) return cap;
		for (int &i = iter[to]; i < graph[to].size(); i++) {
			int next = graph[to][i].to;
			Edge &edge = graph[next][graph[to][i].rev];
			int res;
			if (edge.cap <= EPS || level[next] >= level[to]) continue;
			res = dfs(from, next, min(cap - flow, edge.cap));
			if (res <= 0) continue;
			edge.cap -= res;
			graph[to][i].cap += res;
			flow += res;
			if (abs(flow - cap) <= 0) break;
		}
		return flow;
	}

public:

	MaxFlow(const int n, int inf = 999999999) :
		level(n), iter(n), que(n), size(n), graph(n), INF(inf) {}

	void add_edge(const int from, const int to, const int cap) {
		graph[from].push_back(Edge(to, cap, graph[to].size()));
		graph[to].push_back(Edge(from, 0, graph[from].size() - 1));
	}

	void add_undirected_edge(const int from, const int to, const int cap) {
		graph[from].push_back(Edge(to, cap, graph[to].size()));
		graph[to].push_back(Edge(from, cap, graph[from].size() - 1));
	}

	int max_flow(const int source, const int sink) {
		int flow = 0;

		while (1) {
			bfs(source, sink);

			if (level[sink] == -1) return flow;

			for (int i = 0; i < size; i++) iter[i] = 0;

			flow += dfs(source, sink, INF);
		}
	}
};

//最小コストフロー問題を解く。隣接行列を与えて関数一発で解いてくれるので気軽に使える。
tuple<int, int, vector<vector<int>>>min_cost_flow_(
	const vector<vector<int>>&capacity,
	const vector<vector<int>>&cost,
	const int source,
	const int sink,
	const int upper_limit = 999999999999) {
	//sourceからsinkまで目一杯(ただし最大upper_limit)流した時の
	//総費用、総流量、各辺の流量を求めて返す。

	const int N = capacity.size();
	const int INFINITE = 999999999999;
	typedef array<int, 2> edge;

	rep(i, N)assert(capacity[i].size() == N);
	assert(cost.size() == N);
	rep(i, N)assert(cost[i].size() == N);
	assert(0 <= source && source < N && 0 <= sink && sink < N);
	rep(i, N)rep(j, N)if (0 != capacity[i][j])assert(0 < capacity[i][j] && capacity[j][i] == 0);
	rep(i, N)rep(j, N)assert(cost[i][j] == -cost[j][i]);
	rep(i, N)rep(j, N)if (0 < capacity[i][j])assert(0 <= cost[i][j]);

	vector<vector<int>>flow(N, vector<int>(N, 0));

	int total_cost = 0;
	int total_flow = 0;
	vector<int> potential(N, 0);
	vector<int> previous_vertex(N, 0);
	while (total_flow < upper_limit) {
		priority_queue<edge, vector<edge>, greater<edge>>Dijkstra;
		vector<int>dist(N, INFINITE);
		dist[source] = 0;
		Dijkstra.push({ 0,source });
		while (!Dijkstra.empty()) {
			const edge p = Dijkstra.top(); Dijkstra.pop();
			const int vertex = p[1];
			if (dist[vertex] < p[0])continue;
			rep(i, N)if (capacity[vertex][i] - flow[vertex][i]) {
				const int next_cost = cost[vertex][i];
				if (dist[vertex] + next_cost + potential[vertex] - potential[i] < dist[i]) {
					dist[i] = dist[vertex] + next_cost + potential[vertex] - potential[i];
					previous_vertex[i] = vertex;
					Dijkstra.push({ dist[i],i });
				}
			}
		}
		if (dist[sink] == INFINITE)break;
		rep(i, N)potential[i] += dist[i];

		int new_flow = upper_limit - total_flow;
		for (int vertex = sink; vertex != source; vertex = previous_vertex[vertex]) {
			new_flow = min(new_flow,
				capacity[previous_vertex[vertex]][vertex] -
				flow[previous_vertex[vertex]][vertex]);
		}
		total_flow += new_flow;
		total_cost += new_flow * potential[sink];
		for (int vertex = sink; vertex != source; vertex = previous_vertex[vertex]) {
			flow[previous_vertex[vertex]][vertex] += new_flow;
			flow[vertex][previous_vertex[vertex]] -= new_flow;
		}
	}

	return make_tuple(total_cost, total_flow, flow);
}

int gcd_(int a, int b) {
	if (a < b)swap(a, b);
	if (b == 0)return a;
	return gcd_(b, a % b);
}

int hungarian_(const vector<vector<int>>&cost) {

	//rep(i, cost.size())assert(cost.size() == cost[i].size());

	const int INFINITE = 99999999;

	//http://www.prefield.com/algorithm/math/hungarian.html
	int n = cost.size(), p, q;
	vector<int> fx(n, INFINITE), fy(n, 0);
	vector<int> x(n, -1), y(n, -1);
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			fx[i] = max(fx[i], cost[i][j]);
	for (int i = 0; i < n; ) {
		vector<int> t(n, -1), s(n + 1, i);
		for (p = q = 0; p <= q && x[i] < 0; ++p)
			for (int k = s[p], j = 0; j < n && x[i] < 0; ++j)
				if (fx[k] + fy[j] == cost[k][j] && t[j] < 0) {
					s[++q] = y[j], t[j] = k;
					if (s[q] < 0)
						for (p = j; p >= 0; j = p)
							y[j] = k = t[j], p = x[k], x[k] = j;
				}
		if (x[i] < 0) {
			int d = INFINITE;
			for (int k = 0; k <= q; ++k)
				for (int j = 0; j < n; ++j)
					if (t[j] < 0) d = min(d, fx[s[k]] + fy[j] - cost[s[k]][j]);
			for (int j = 0; j < n; ++j) fy[j] += (t[j] < 0 ? 0 : d);
			for (int k = 0; k <= q; ++k) fx[s[k]] -= d;
		}
		else ++i;
	}
	int ret = 0;
	for (int i = 0; i < n; ++i) ret += cost[i][x[i]];
	return ret;
}

template<int DIMENSION>class fminsearch {
	//使い方
	//1: 次数DIMENSIONのdouble型ベクトルを引数として、
	//   double型スカラーを返すようなラムダ式λを用意する。これ目的関数とする。
	//2: このクラスのコンストラクタ引数としてλを渡す。
	//3: このクラスのメンバ関数のrun()を呼ぶ。
	//   返り値は、それをλに入れると最小値を返すようなベクトルである。
	//内部実装
	//   Nelder-Mead法(a.k.a.アメーバ法,滑降シンプレックス法)で最小値を探索する。

private:
	typedef std::array<double, DIMENSION> vec_type;
	typedef std::array<std::pair<double, vec_type>, DIMENSION + 1> amoeba_type;
	typedef std::function<double(const vec_type)> func_type;

	func_type lambda_exp;

	bool early_termination(
		const double TolFun,
		const double TolX,
		const amoeba_type &vecs) {

		double maxnum, minnum;
		maxnum = vecs[DIMENSION].first;
		minnum = vecs[DIMENSION].first;
		rep(d, DIMENSION) {
			maxnum = std::max(maxnum, vecs[d].first);
			minnum = std::min(minnum, vecs[d].first);
		}
		if (TolFun <= maxnum - minnum)return false;
		rep(i, DIMENSION) {
			maxnum = vecs[DIMENSION].second[i];
			minnum = vecs[DIMENSION].second[i];
			rep(d, DIMENSION) {
				maxnum = std::max(maxnum, vecs[d].second[i]);
				minnum = std::min(minnum, vecs[d].second[i]);
			}
			if (TolX <= maxnum - minnum)return false;
		}
		return true;
	}

public:
	fminsearch(const func_type func) { this->lambda_exp = func; }

	double eval(vec_type x) { return lambda_exp(x); }

	vec_type run(
		const vec_type InitVec,
		const int MaxIter = 200 * DIMENSION,
		const double TolFun = 1e-10,
		const double TolX = 1e-10) {

		constexpr double ALPHA = 1.0;
		constexpr double GAMMA = 2.0;
		constexpr double RHO = -0.5;
		constexpr double SIGMA = 0.5;

		amoeba_type vecs;
		rep(i, DIMENSION + 1)vecs[i].second = InitVec;
		rep(i, DIMENSION)vecs[i].second[i] += 1.0;
		rep(i, DIMENSION + 1)vecs[i].first = eval(vecs[i].second);

		rep(mr, MaxIter) {
			if (early_termination(TolFun, TolX, vecs))break;
			std::sort(ALL(vecs));
			vec_type x0 = { 0.0 }, xr = { 0.0 };
			rep(i, DIMENSION)rep(j, DIMENSION)x0[j] += vecs[i].second[j];
			rep(i, DIMENSION)x0[i] /= (double)DIMENSION;
			rep(i, DIMENSION)xr[i] = x0[i] + ALPHA * (x0[i] - vecs[DIMENSION].second[i]);
			const double fx0 = eval(x0);
			const double fxr = eval(xr);
			if (vecs[0].first <= fxr && fxr < vecs[DIMENSION].first) {
				//反射
				vecs[DIMENSION] = mp(fxr, xr);
			}
			else if (fxr < vecs[0].first) {
				//膨張
				vec_type xe = { 0.0 };
				rep(i, DIMENSION)xe[i] = x0[i] + GAMMA * (x0[i] - vecs[DIMENSION].second[i]);
				const double fxe = eval(xe);
				vecs[DIMENSION] = fxe < fxr ? mp(fxe, xe) : mp(fxr, xr);
			}
			else {
				//収縮
				vec_type xc = { 0.0 };
				rep(i, DIMENSION)xc[i] = x0[i] + RHO * (x0[i] - vecs[DIMENSION].second[i]);
				const double fxc = eval(xc);
				if (fxc < vecs[DIMENSION].first) vecs[DIMENSION] = mp(fxc, xc);
				else {
					REP(i, 1, DIMENSION + 1)rep(j, DIMENSION) {
						vecs[i].second[j] = vecs[0].second[j] +
							SIGMA * (vecs[i].second[j] - vecs[0].second[j]);
					}
					REP(i, 1, DIMENSION + 1)vecs[i].first = eval(vecs[i].second);
				}
			}
		}
		std::sort(ALL(vecs));
		return vecs[0].second;
	}
};

//エラトステネスの篩で高速に素数生成して、素数かどうか聞くクエリに答える。
template<int MAX>class Eratosthenes {
private:
	const int LENGTH = (MAX / 64) + 1;
	const uint64_t one = 1;
	vector<uint64_t>isPrimeTable;

public:
	Eratosthenes() :isPrimeTable(LENGTH, 0) {
		vector<uint64_t>isKnownTable(LENGTH, 0);
		for (int i = 2; i <= MAX; i++)
		{
			if (!(isKnownTable[i / 64] & (one << (i % 64)))) {
				isKnownTable[i / 64] |= (one << (i % 64));
				isPrimeTable[i / 64] |= (one << (i % 64));
				for (int j = i * 2; j <= MAX; j += i) {
					isKnownTable[j / 64] |= (one << (j % 64));
				}
			}
		}
	}
	bool isPrime(const int num) {
		return (isPrimeTable[num / 64] & (one << (num % 64))) ? true : false;
	}
};

//気軽に使える素数判定。
bool isPrime(int a) {
	for (int i = 2; i <= sqrt(a + 1); ++i) {
		if (a%i == 0) {
			return 0;
		}
	}
	return 1;
}

//Suffix ArrayをSAISでO(N)で計算してくれる。LCP Arrayも計算してくれる。
class SuffixArrayMaker {

private:

	bool TGet(const std::vector<uint8_t>&t, const int index) {
		return (t[index / 8] & (uint8_t(1) << (index % 8))) ? true : false;
	}
	void TSet(std::vector<uint8_t>&t, const int index, const bool value) {
		t[index / 8] = value ?
			(t[index / 8] | (uint8_t(1) << (index % 8))) :
			(t[index / 8] & (~(uint8_t(1) << (index % 8))));
	}
	bool IsLMS(const std::vector<uint8_t>&t, const int index) {
		return (index > 0) && TGet(t, index) && (!TGet(t, index - 1));
	}
	template<typename T>void GetBuckets(
		const T* input,
		std::vector<int>&bucket,
		const int n,
		const int K,
		const bool end) {
		for (int i = 0; i <= K; ++i)bucket[i] = 0;
		for (int i = 0; i < n; ++i)++bucket[input[i]];
		int sum = 0;
		for (int i = 0; i <= K; ++i) {
			sum += bucket[i];
			bucket[i] = end ? sum : (sum - bucket[i]);
		}
	}
	template<typename T>void InducedSortArrayL(
		const T* input,
		const std::vector<uint8_t>&t,
		int* suffix_array,
		std::vector<int>&bucket,
		const int n,
		const int K,
		const bool end) {
		GetBuckets<T>(input, bucket, n, K, end);
		for (int i = 0; i < n; ++i) {
			const int j = suffix_array[i] - 1;
			if ((j >= 0) && (!TGet(t, j)))suffix_array[bucket[input[j]]++] = j;
		}
	}
	template<typename T>void InducedSortArrayS(
		const T* input,
		const std::vector<uint8_t>&t,
		int* suffix_array,
		std::vector<int>&bucket,
		const int n,
		const int K,
		const bool end) {
		GetBuckets<T>(input, bucket, n, K, end);
		for (int i = n - 1; i >= 0; --i) {
			const int j = suffix_array[i] - 1;
			if ((j >= 0) && TGet(t, j))suffix_array[--bucket[input[j]]] = j;
		}
	}
	template<typename T>void SA_IS(const T* input, int* suffix_array, const int n, const int K) {

		std::vector<uint8_t>t((n + 7) / 8, 0);
		TSet(t, n - 2, false);
		TSet(t, n - 1, true);
		for (int i = n - 3; i >= 0; --i) {
			bool sets = (input[i] < input[i + 1]) ||
				(input[i] == input[i + 1] && TGet(t, i + 1));
			TSet(t, i,
				(input[i] < input[i + 1]) ||
				(input[i] == input[i + 1] && TGet(t, i + 1))
			);
		}

		//stage 1
		std::vector<int>bucket(K + 1, 0);
		GetBuckets<T>(input, bucket, n, K, true);
		for (int i = 0; i < n; ++i)suffix_array[i] = -1;
		for (int i = 1; i < n; ++i)if (IsLMS(t, i))suffix_array[--bucket[input[i]]] = i;
		InducedSortArrayL<T>(input, t, suffix_array, bucket, n, K, false);
		InducedSortArrayS<T>(input, t, suffix_array, bucket, n, K, true);
		int n1 = 0;
		for (int i = 0; i < n; ++i) {
			if (IsLMS(t, suffix_array[i]))suffix_array[n1++] = suffix_array[i];
		}
		for (int i = n1; i < n; ++i)suffix_array[i] = -1;
		int name = 0, prev = -1;
		for (int i = 0; i < n1; ++i) {
			int pos = suffix_array[i];
			bool diff = false;
			for (int d = 0; d < n; ++d) {
				if ((prev == -1) ||
					(input[pos + d] != input[prev + d]) ||
					(TGet(t, pos + d) != TGet(t, prev + d))) {
					diff = true;
					break;
				}
				else if (d > 0 && (IsLMS(t, pos + d) || IsLMS(t, prev + d)))break;
			}
			if (diff) {
				name++;
				prev = pos;
			}
			pos = ((pos % 2) == 0) ? (pos / 2) : ((pos - 1) / 2);
			suffix_array[n1 + pos] = name - 1;
		}
		for (int i = n - 1, j = n - 1; i >= n1; --i) {
			if (suffix_array[i] >= 0)suffix_array[j--] = suffix_array[i];
		}

		//stage 2
		int* suffix_array1 = suffix_array;
		int* s1 = suffix_array + n - n1;
		if (name < n1) {
			SA_IS<int>(s1, suffix_array1, n1, name - 1);
		}
		else {
			for (int i = 0; i < n1; ++i)suffix_array1[s1[i]] = i;
		}

		//stage 3
		GetBuckets<T>(input, bucket, n, K, true);
		for (int i = 1, j = 0; i < n; ++i) {
			if (IsLMS(t, i))s1[j++] = i;
		}
		for (int i = 0; i < n1; ++i)suffix_array1[i] = s1[suffix_array1[i]];
		for (int i = n1; i < n; ++i)suffix_array[i] = -1;
		for (int i = n1 - 1; i >= 0; --i) {
			const int j = suffix_array[i];
			suffix_array[i] = -1;
			suffix_array[--bucket[input[j]]] = j;
		}
		InducedSortArrayL<T>(input, t, suffix_array, bucket, n, K, false);
		InducedSortArrayS<T>(input, t, suffix_array, bucket, n, K, true);
	}

public:

	std::vector<int> SuffixArrayConstruction(const std::vector<uint8_t>& input) {

		for (int i = 0; i < input.size() - 1; ++i)assert(input[i] != 0);
		assert(input.back() == 0);

		std::vector<int>suffix_array(input.size(), 0);
		SA_IS<uint8_t>(&input[0], &suffix_array[0], input.size(), 256);
		return suffix_array;
	}

	std::vector<int> SuffixArrayConstruction(const std::string& input) {
		std::vector<uint8_t>tmp;
		for (char c : input)tmp.push_back(uint8_t(c));
		tmp.push_back(0);
		return SuffixArrayConstruction(tmp);
	}

	//LCP arrayとは、suffix array上で隣接する接尾辞の共通する接頭辞の文字数を格納した配列で、長さはN-1である。
	//Kasaiの方法でO(N)で計算する
	//Suffix Arrayと密接に関係しているのでこのクラスの中に入れた。
	std::vector<int>KasaiLCPArrayConstruction(
		const std::vector<uint8_t>& input,
		const std::vector<int>& suffix_array) {

		for (int i = 0; i < input.size() - 1; ++i)assert(input[i] != 0);
		assert(input.back() == 0);
		assert(input.size() == suffix_array.size());

		std::vector<int>inverse_suffix_array(suffix_array.size(), 0);
		for (int i = 0; i < inverse_suffix_array.size(); ++i) {
			inverse_suffix_array[suffix_array[i]] = i;
		}

		std::vector<int>lcp_array(input.size() - 1, 0);
		int lcp = 0;
		for (int i = 0; i < input.size(); ++i) {
			if (inverse_suffix_array[i] == input.size() - 1) {
				lcp = 0;
				continue;
			}
			const int pos1 = suffix_array[inverse_suffix_array[i]];
			const int pos2 = suffix_array[inverse_suffix_array[i] + 1];
			while (input[pos1 + lcp] == input[pos2 + lcp])lcp++;
			lcp_array[inverse_suffix_array[i]] = lcp;
			if (lcp > 0)--lcp;
		}
		return lcp_array;
	}

	std::vector<int>KasaiLCPArrayConstruction(
		const std::string& input,
		const std::vector<int>& suffix_array) {

		std::vector<uint8_t>tmp;
		for (char c : input)tmp.push_back(uint8_t(c));
		tmp.push_back(0);
		return KasaiLCPArrayConstruction(tmp, suffix_array);
	}
};

//数列を圧縮して持っておいて、参照クエリを受ける。
//その数列は[0,N)を並べ替えたものであることを仮定しており、
//かつ一様ランダムな並べ替えではないことを想定して圧縮する。
template<typename BlockType> class VerticalCodePermutation {
private:

	static_assert(
		is_same<BlockType, uint64_t>::value ||
		is_same<BlockType, uint32_t>::value ||
		is_same<BlockType, uint16_t>::value ||
		is_same<BlockType, uint8_t>::value);

	static constexpr int block = sizeof(BlockType) * 8;
	uint64_t inputsize;
	std::vector<uint64_t>psi;
	std::vector<BlockType>code;
	std::vector<uint64_t>code_index;

	static uint64_t popcount(uint64_t x) {
		x = (x & 0x5555555555555555ULL) + ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
		x = (x & 0x3333333333333333ULL) + ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
		x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
		x *= 0x0101010101010101ULL;
		return x >> 56;
	}

	//x以上の2べき乗数のうち最小の数が2^kとしてkを返す。
	static int64_t log2_ceiling(uint64_t x) {

		if (x <= 0)return 0;
		x--;
		//msbの位置を求めるために、ビットを下位に伝播させてpopcntする。
		for (int i = 0; i < 6; ++i)x |= x >> (1ULL << i);
		return popcount(x);
	}

public:

	VerticalCodePermutation(const std::vector<uint64_t>& input) {
		//inputが元の数列だとして、差分化してvertical codeで保持する。
		//inputの各要素は[0,N)の並べ替えであることを仮定する。

		{
			auto testarray = input;
			sort(testarray.begin(), testarray.end());
			for (uint64_t i = 0; i < testarray.size(); ++i)assert(testarray[i] == i);
		}

		inputsize = input.size();
		const int64_t code_length = (input.size() + block - 1) / block;

		//psi[i]には、0～(i-1)ブロック内の差分列の総和をsizeで割った余りを格納する。
		//「差分列の総和」は元の数列の値とは異なる。なぜなら、
		//ここで言う「差分列」は、圧縮しやすい単調増加列にするため、差分に-1や+Nしているからである。
		psi = std::vector<uint64_t>(code_length, 0);
		//for (int64_t i = 1; i < code_length; ++i)psi[i] = input[i * block - 1];

		code.clear();
		code_index = std::vector<uint64_t>(code_length + 1, 0);

		for (int64_t b = 0; b < code_length; ++b) {
			const int64_t offset = b * block;

			//offset番目からblock個について、単調増加な差分列を作る。
			std::vector<uint64_t>diff(std::min(int64_t(block), int64_t(input.size() - offset)), 0);
			uint64_t diffmax = 0;
			for (int64_t i = 0; i < diff.size(); ++i) {
				const int64_t j = offset + i;
				if (j == 0)diff[i] = input[j];
				else if (input[j - 1] < input[j])diff[i] = input[j] - input[j - 1] - 1;
				else {
					//↓はinputが[0,N)の並べ替えであれば必ず成り立つ。
					//assert(input[j - 1] - input[j] < input.size());
					diff[i] = input[j] - input[j - 1] - 1 + input.size();
				}
				diffmax = std::max(diffmax, diff[i]);
			}

			//作った差分列をcodeの末尾に追加格納する。
			const int64_t bit_demand = log2_ceiling(diffmax + 1);
			code.resize(code.size() + bit_demand);
			code_index[b + 1] = code.size();
			for (int i = 0; i < diff.size(); ++i) {
				for (uint64_t num = diff[i], j = code_index[b]; num; num /= 2, ++j) {
					code[j] |= (num & uint64_t(1)) << i;
				}
			}

			//phiの最後に、ここまでの差分列の総和を格納する。
			if (b < code_length - 1) {
				psi[b + 1] = psi[b];
				for (const int x : diff)psi[b + 1] += x;
				psi[b + 1] %= input.size();
			}
		}
	}

	VerticalCodePermutation() {
		VerticalCodePermutation(std::vector<uint64_t>{0});
	}

	uint64_t Get(const int64_t index) {
		const int64_t b = index / block;
		const int64_t bits = index - (b * block);

		const BlockType bitmask = BlockType(0xFFFF'FFFF'FFFF'FFFFULL) >> ((block - 1) - bits);

		uint64_t answer = psi[b];
		for (uint64_t i = code_index[b]; i < code_index[b + 1]; ++i) {
			answer += popcount(code[i] & bitmask) << (i - code_index[b]);
		}
		return (answer + index) % inputsize;
	}

};

//数列を圧縮して持っておいて、参照クエリを受ける。
//その数列は広義単調増加列であることを仮定している。
template<typename BlockType> class VerticalCodeWeaklyIncrease {
private:

	static_assert(
		is_same<BlockType, uint64_t>::value ||
		is_same<BlockType, uint32_t>::value ||
		is_same<BlockType, uint16_t>::value ||
		is_same<BlockType, uint8_t>::value);

	static constexpr int block = sizeof(BlockType) * 8;
	uint64_t inputsize;
	std::vector<uint64_t>psi;
	std::vector<BlockType>code;
	std::vector<uint64_t>code_index;

	static uint64_t popcount(uint64_t x) {
		x = (x & 0x5555555555555555ULL) + ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
		x = (x & 0x3333333333333333ULL) + ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
		x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
		x *= 0x0101010101010101ULL;
		return x >> 56;
	}

	//x以上の2べき乗数のうち最小の数が2^kとしてkを返す。
	static int64_t log2_ceiling(uint64_t x) {

		if (x <= 0)return 0;
		x--;
		//msbの位置を求めるために、ビットを下位に伝播させてpopcntする。
		for (int i = 0; i < 6; ++i)x |= x >> (1ULL << i);
		return popcount(x);
	}

public:

	VerticalCodeWeaklyIncrease(const std::vector<uint64_t>& input) {
		//inputが元の数列だとして、差分化してvertical codeで保持する。
		//inputは広義単調増加列であることを仮定している。

		for (uint64_t i = 1; i < input.size(); ++i)assert(input[i - 1] <= input[i]);

		inputsize = input.size();
		const int64_t code_length = (input.size() + block - 1) / block;

		//psi[i]には、0～(i-1)ブロック内の差分列の総和、つまり直前の項の値を格納する。
		psi = std::vector<uint64_t>(code_length, 0);
		for (int64_t i = 1; i < code_length; ++i)psi[i] = input[i * block - 1];

		code.clear();
		code_index = std::vector<uint64_t>(code_length + 1, 0);

		for (int64_t b = 0; b < code_length; ++b) {
			const int64_t offset = b * block;

			//offset番目からblock個について、差分列を作る。
			std::vector<uint64_t>diff(std::min(int64_t(block), int64_t(input.size() - offset)), 0);
			uint64_t diffmax = 0;
			for (int64_t i = 0; i < diff.size(); ++i) {
				const int64_t j = offset + i;
				if (j == 0)diff[i] = input[j];
				else diff[i] = input[j] - input[j - 1];
				diffmax = std::max(diffmax, diff[i]);
			}

			//作った差分列をcodeの末尾に追加格納する。
			const int64_t bit_demand = log2_ceiling(diffmax + 1);
			code.resize(code.size() + bit_demand);
			code_index[b + 1] = code.size();
			for (int i = 0; i < diff.size(); ++i) {
				for (uint64_t num = diff[i], j = code_index[b]; num; num /= 2, ++j) {
					code[j] |= (num & uint64_t(1)) << i;
				}
			}
		}
	}

	VerticalCodeWeaklyIncrease() {
		VerticalCodeWeaklyIncrease(std::vector<uint64_t>{0});
	}

	uint64_t Get(const int64_t index) {
		const int64_t b = index / block;
		const int64_t bits = index - (b * block);

		const BlockType bitmask = BlockType(0xFFFF'FFFF'FFFF'FFFFULL) >> ((block - 1) - bits);

		uint64_t answer = psi[b];
		for (uint64_t i = code_index[b]; i < code_index[b + 1]; ++i) {
			answer += popcount(code[i] & bitmask) << (i - code_index[b]);
		}
		return answer;
	}

};

template<uint64_t bit_length>class NBitArray {

private:

	//bit_length==0は実行時モードで、実行時に引数で与えるが
	//即値でないことで非効率になりうる
	static_assert((1 <= bit_length && bit_length < 64) || bit_length == 0);

	std::vector<uint64_t>B;
	uint64_t arraysize;

	uint64_t my_bit_length;

public:

	NBitArray(const uint64_t size) {
		assert(bit_length != 0);
		const uint64_t pack = 64 / (bit_length ? bit_length : 1);
		B = std::vector<uint64_t>((size + pack - 1) / pack);
		arraysize = size;
	}
	NBitArray(const uint64_t size, const uint64_t length) {
		assert(bit_length == 0);
		assert(1 <= length && length <= 64);
		my_bit_length = length;
		const uint64_t pack = 64 / my_bit_length;
		B = std::vector<uint64_t>((size + pack - 1) / pack);
		arraysize = size;
	}

	NBitArray() {
		if (bit_length)NBitArray(1);
		else NBitArray(1, 1);
	}

	void Set(const uint64_t index, const uint64_t number) {

		if (bit_length == 8) {
			uint8_t* p = (uint8_t*)(&B[0]);
			p[index] = number;
			return;
		}

		const uint64_t pack = 64 / (bit_length ? bit_length : my_bit_length);
		assert(index < arraysize);
		assert(number < (1ULL << (bit_length ? bit_length : my_bit_length)));
		const uint64_t internal_index = index / pack;
		const uint64_t internal_pos = index % pack;
		const uint64_t offset = internal_pos * (bit_length ? bit_length : my_bit_length);
		const uint64_t bitmask = ((1ULL << (bit_length ? bit_length : my_bit_length)) - 1ULL) << offset;
		B[internal_index] &= ~bitmask;
		B[internal_index] |= number << offset;
	}

	uint64_t Get(const uint64_t index) {

		if (bit_length == 8) {
			uint8_t* p = (uint8_t*)(&B[0]);
			uint8_t x = p[index];
			return (uint64_t)(x);
		}

		const uint64_t pack = 64 / (bit_length ? bit_length : my_bit_length);
		assert(index < arraysize);
		const uint64_t internal_index = index / pack;
		const uint64_t internal_pos = index % pack;
		const uint64_t offset = internal_pos * (bit_length ? bit_length : my_bit_length);
		const uint64_t bitmask = (1ULL << (bit_length ? bit_length : my_bit_length)) - 1ULL;
		return (B[internal_index] >> offset) & bitmask;
	}

	uint64_t size() {
		return arraysize;
	}
};
//ビット列を持っておいて、参照クエリとRank/Selectクエリを受ける。
//入力されたビット列はこれ以上圧縮できないと想定し、そのまま保存する。
template<uint64_t global>class BitVector {

private:

	//template引数globalは補助データ構造の密度を指定する。
	//global==0はおまかせモードで、入力データの大きさから最適な密度を求めて用いるが
	//即値でないことで非効率になりうる
	static_assert((8 <= global && global <= 16) || global == 0);

	std::vector<uint64_t>B;
	std::vector<uint64_t>rank_global;
	NBitArray<global>rank_local;

	uint64_t my_global;

	static uint64_t popcount(uint64_t x) {
		x = (x & 0x5555555555555555ULL) + ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
		x = (x & 0x3333333333333333ULL) + ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
		x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
		x *= 0x0101010101010101ULL;
		return x >> 56;
	}

public:

	//inputがビット列だと解釈して、rank/selectのための追加データ構造を構築する。
	BitVector(const std::vector<uint64_t>& input) {

		assert(input.size() >= 1);

		B = input;

		my_global = median_constexpr(8, 16, log2_ceiling(uint64_t(std::pow(std::log2(input.size() * 64), 2.0) + 0.5)));

		//rank_global[i]の末尾以外には、[0,i*global_length)の範囲に立っているビットの数を格納する。
		//末尾にはデータ全体で立っているビットの数を格納する。
		const uint64_t global_length = 1ULL << (global ? global : my_global);
		const uint64_t global_interval = global_length / 64;
		rank_global = std::vector<uint64_t>(((input.size() + global_interval - 1) / global_interval) + 1);

		//rank_local[i]には、[i*global_interval,(1<<local)*i)の範囲に立っているビットの数を格納する。
		//rank_localの要素数はBの要素数とほぼ同じだが、rank_globalが末尾で剰余部分を多く確保している場合はそれに準ずる。
		if (global) {
			rank_local = NBitArray<global>(((input.size() + global_interval - 1) / global_interval) * global_interval);
		}
		else {
			rank_local = NBitArray<global>(((input.size() + global_interval - 1) / global_interval) * global_interval, my_global);
		}

		uint64_t global_sum = 0;
		uint64_t local_sum = 0;
		for (int64_t i = 0; i < input.size(); ++i) {
			if (i % global_interval == 0) {
				local_sum = 0;
			}
			rank_local.Set(i, local_sum);
			const uint64_t pop = popcount(input[i]);
			global_sum += pop;
			local_sum += pop;
			if (i % global_interval == global_interval - 1) {
				rank_global[i / global_interval + 1] = global_sum;
			}
		}
		rank_global.back() = global_sum;

		for (int64_t i = input.size(); i < rank_local.size(); ++i) {
			rank_local.Set(i, rank_global.back() - rank_global[rank_global.size() - 2]);
		}
	}

	BitVector() {
		BitVector(std::vector<uint64_t>{0});
	}

	bool access(const uint64_t index) {
		if (index >= B.size() * 64)return false;
		return (B[index / 64] & (uint64_t(1) << (index % 64))) != 0;
	}

	uint64_t rank1(const uint64_t index) {
		//[0,index)にいくつ立っているビットがあるか求めて返す。
		if (index >= B.size() * 64)return rank_global.back();

		uint64_t answer = rank_global[index / (1ULL << (global ? global : my_global))] + rank_local.Get(index / 64);
		answer += popcount(B[index / 64] & ((uint64_t(1) << (index % 64)) - 1));
		return answer;
	}
	uint64_t rank0(const uint64_t index) {
		//[0,index)にいくつ立っていないビットがあるか求めて返す。
		if (index >= B.size() * 64)return B.size() * 64 - rank_global.back();
		return index - rank1(index);
	}

	uint64_t select1(const uint64_t count) {
		//count番目(0-origin)の1のビットの位置を求めて返す。低速。

		if (rank_global.back() <= count)return std::numeric_limits<uint64_t>::max();

		const uint64_t global_length = 1ULL << (global ? global : my_global);
		const uint64_t global_interval = global_length / 64;

		//global表の上で二分探索する。
		uint64_t lb = 0, ub = rank_global.size();
		while (lb + 1 < ub) {
			const uint64_t mid = (lb + ub) / 2;
			if (count < rank_global[mid])ub = mid;
			else lb = mid;
		}

		//この時点で、欲しい位置は[lb*global_length,(lb+1)*global_length)のどこかにある。

		//local表の上を二分探索する。
		uint64_t x = rank_global[lb];
		uint64_t lb2 = 0, ub2 = global_interval;
		while (lb2 + 1 < ub2) {
			const uint64_t mid = (lb2 + ub2) / 2;
			if (count < x + rank_local.Get(lb * global_interval + mid)) {
				ub2 = mid;
			}
			else {
				lb2 = mid;
			}
		}
		x += rank_local.Get(lb * global_interval + lb2);
		lb = lb * global_length + 64 * lb2;

		//この時点で、欲しい位置は[lb,lb+64)のどこかにある。
		//[0,lb)で立っているビットの数はxである。

		const uint64_t index = lb / 64;

		//B[index]の64bitのうち、
		//i番目のビットが立っていて、かつ[0,i)の中で(count-x)個のビットが立っているようなiを求め、
		//(lb+i)を答えとして返す。

		//TODO: もっと速くてかっこいいビット演算があるはず……bsfとか使って
		for (uint64_t i = 0; i < 64; ++i) {
			if (B[index] & (1ULL << i)) {
				if (count == x++)return lb + i;
			}
		}

		assert(0);
		return -1;
	}

	uint64_t select0(const uint64_t count) {
		//count番目(0-origin)の0のビットの位置を求めて返す。低速。

		if (B.size() * 64 - rank_global.back() <= count)return std::numeric_limits<uint64_t>::max();

		const uint64_t global_length = 1ULL << (global ? global : my_global);
		const uint64_t global_interval = global_length / 64;

		//global表の上で二分探索する。
		uint64_t lb = 0, ub = rank_global.size();
		while (lb + 1 < ub) {
			const uint64_t mid = (lb + ub) / 2;
			if (count < mid * global_length - rank_global[mid])ub = mid;
			else lb = mid;
		}

		//この時点で、欲しい位置は[lb*global_length,(lb+1)*global_length)のどこかにある。

		//local表の上を二分探索する。
		uint64_t x = lb * global_length - rank_global[lb];
		uint64_t lb2 = 0, ub2 = global_interval;
		while (lb2 + 1 < ub2) {
			const uint64_t mid = (lb2 + ub2) / 2;
			if (count < x + 64 * mid - rank_local.Get(lb * global_interval + mid)) {
				ub2 = mid;
			}
			else {
				lb2 = mid;
			}
		}
		x += 64 * lb2 - rank_local.Get(lb * global_interval + lb2);
		lb = lb * global_length + 64 * lb2;

		//この時点で、欲しい位置は[lb,lb+64)のどこかにある。
		//[0,lb)でゼロのビットの数はxである。

		const uint64_t index = lb / 64;

		//B[index]の64bitのうち、
		//i番目のビットがゼロて、かつ[0,i)の中で(count-x)個のビットがゼロであるようなiを求め、
		//(lb+i)を答えとして返す。

		//TODO: もっと速くてかっこいいビット演算があるはず……bsfとか使って
		for (uint64_t i = 0; i < 64; ++i) {
			if (!(B[index] & (1ULL << i))) {
				if (count == x++)return lb + i;
			}
		}

		assert(0);
		return -1;
	}

	uint64_t sum1() { return rank_global.back(); }

};
template<uint64_t sparseness> class BitVectorSparse {

private:

	//1/(2^sparseness)の確率でビットが立っているときにうまく機能する。
	//sparseness==0はおまかせモードで、入力データの大きさから最適な密度を求めて用いるが
	//即値でないことで非効率になりうる
	static_assert((1 <= sparseness && sparseness <= 32) || sparseness == 0);

	NBitArray<sparseness>position_least;
	BitVector<0> position_most;
	uint64_t inputsizex64, unarysize, my_sparseness;

	static uint64_t popcount(uint64_t x) {
		x = (x & 0x5555555555555555ULL) + ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
		x = (x & 0x3333333333333333ULL) + ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
		x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
		x *= 0x0101010101010101ULL;
		return x >> 56;
	}

public:

	//inputがビット列だと解釈して、access/rank/selectのためのデータ構造を構築する。
	BitVectorSparse(const std::vector<uint64_t>& input) {

		assert(input.size() >= 1);

		//1が立っているビットの位置を全部数える。
		std::vector<uint64_t>position;
		for (uint64_t i = 0; i < input.size(); ++i) {
			for (uint64_t j = 0; j < 64; ++j) {
				if (input[i] & (1ULL << j))position.push_back(i * 64 + j);
			}
		}

		if (position.size() == 0) {
			unarysize = 0;
			return;
		}

		my_sparseness = median_constexpr(1, 32, log2_ceiling(uint64_t(std::log2(input.size() * 64 / position.size()) + 0.5)));

		//positionの下位sparsenessビットはそのまま保存する。

		if (sparseness) {
			position_least = NBitArray<sparseness>(position.size());
		}
		else {
			position_least = NBitArray<sparseness>(position.size(), my_sparseness);
		}

		for (uint64_t i = 0; i < position.size(); ++i) {
			position_least.Set(i, position[i] & ((1ULL << (sparseness ? sparseness : my_sparseness)) - 1));
		}


		//positionの上位ビットは広義単調増加列だが、差分をunary符号で保存する。
		for (uint64_t i = 0; i < position.size(); ++i)position[i] >>= (sparseness ? sparseness : my_sparseness);

		//このunary符号の中には、position.size()個の1とposition_mostの個数個の0が含まれる。
		const uint64_t unary_one = position.size();
		const uint64_t unary_zero = (input.size() * 64 + (1ULL << (sparseness ? sparseness : my_sparseness)) - 1ULL) / (sparseness ? sparseness : my_sparseness);
		std::vector<uint64_t> unary((unary_one + unary_zero + 63) / 64, 0);
		uint64_t bit = 0;
		for (uint64_t i = 0; i < position[0]; ++i)++bit;
		unary[bit / 64] |= 1ULL << (bit % 64);
		++bit;
		for (uint64_t i = 1; i < position.size(); ++i) {
			const uint64_t diff = position[i] - position[i - 1];
			for (uint64_t j = 0; j < diff; ++j)++bit;
			unary[bit / 64] |= 1ULL << (bit % 64);
			++bit;
		}

		position_most = BitVector<0>(unary);

		inputsizex64 = input.size() * 64;
		unarysize = position.size() + position.back();
	}

	BitVectorSparse() {
		BitVectorSparse(std::vector<uint64_t>{0});
	}

	uint64_t select1(const uint64_t count) {
		//count番目(0-origin)の1のビットの位置を求めて返す。

		if (unarysize == 0 || position_most.sum1() <= count)return std::numeric_limits<uint64_t>::max();

		return ((position_most.select1(count) - count) << (sparseness ? sparseness : my_sparseness)) + position_least.Get(count);
	}

	uint64_t rank1(const uint64_t index) {
		//[0,index)にいくつ立っているビットがあるか求めて返す。
		if (index >= inputsizex64)return position_most.sum1();
		if (unarysize == 0)return 0;

		const uint64_t indmost = index >> (sparseness ? sparseness : my_sparseness);

		//cf.岡之原本p51
		//unary符号において0が(index >> sparseness)回出てきた直後というのは、すなわち元のbitvector上で
		//[((index >> sparseness) << sparseness),inputsize)上で最初に1が出る位置の符号化を意味する。

		uint64_t t = indmost == 0 ? 0 : (position_most.select0(indmost - 1) + 1);

		//unary符号上で[0,t)の中に1が出てきた回数が、すなわち元のbitvector上で
		//[0,((index >> sparseness) << sparseness))中で1が出てきた回数を意味する。これをxに格納する。
		//[0,t)の中に0が出てきた回数は(index >> sparseness)回である(select0をしているので明らか)からこれを引けば良い。
		uint64_t answer = t - (indmost);

		//[((index >> sparseness) << sparseness),index)上で立っているビットがいくつあるかをこれから数える。

		for (uint64_t candidate = answer; t < unarysize && position_most.access(t); candidate++, t++) {
			const uint64_t position = ((index >> (sparseness ? sparseness : my_sparseness)) << (sparseness ? sparseness : my_sparseness)) + position_least.Get(answer);
			if (position >= index)break;
			++answer;
		}

		return answer;
	}

	uint64_t rank0(const uint64_t index) {
		//[0,index)にいくつ立っていないビットがあるか求めて返す。
		if (index >= inputsizex64)return inputsizex64 - position_most.sum1();
		return index - rank1(index);
	}

	bool access(const uint64_t index) {
		//index番目のビットが立っているか調べる。

		if (unarysize == 0)return false;

		const uint64_t leftmost = select1(0);
		if (index == leftmost)return true;
		if (index < leftmost)return false;

		const uint64_t rightmost = select1(position_most.sum1() - 1);
		if (index == rightmost)return true;
		if (index > rightmost)return false;

		//以降はrank1と同じような処理
		const uint64_t indmost = index >> (sparseness ? sparseness : my_sparseness);
		uint64_t t = indmost == 0 ? 0 : (position_most.select0(indmost - 1) + 1);
		uint64_t x = t - (indmost);
		for (uint64_t candidate = x; t < unarysize && position_most.access(t); candidate++, t++) {
			const uint64_t position = ((index >> (sparseness ? sparseness : my_sparseness)) << (sparseness ? sparseness : my_sparseness)) + position_least.Get(x);
			if (position == index)return true;
			if (position > index)return false;
			++x;
		}

		return false;
	}

	uint64_t select0(const uint64_t count) {
		//count番目(0-origin)の0のビットの位置を求めて返す。

		if (inputsizex64 - position_most.sum1() <= count)return std::numeric_limits<uint64_t>::max();
		if (unarysize == 0)return count;

		uint64_t lb = 0, ub = inputsizex64;
		while (lb + 1 < ub) {
			const uint64_t mid = (lb + ub) / 2;
			const uint64_t x = rank0(mid);
			const uint64_t b = access(mid);

			if (x <= count)lb = mid;
			else ub = mid;
		}

		return lb;

		assert(0);
		return -1;
	}

	uint64_t sum1() { return position_most.sum1(); }

};
template<uint64_t block_length> class BitVectorRRR {

private:

	//block_lengthは補助データ構造の密度で、入力ビット列の長さnのときlog2(n)/2にすると効率が良い。
	//block_length==0はおまかせモードで、入力データの大きさから最適なblock_lengthを求めて用いるが
	//即値でないことで非効率になりうる
	static_assert((4 <= block_length && block_length <= 20) || block_length == 0);

	NBitArray<block_length ? log2_ceiling_constexpr(block_length + 1) : 0>block_class;
	BitVector<block_length ? median_constexpr(8, 16, log2_ceiling_constexpr((block_length * block_length * 4) * 2 / 3)) : 0>block_offset;
	std::vector<uint64_t>rank_global;
	std::vector<uint64_t>offset_global;
	BitVectorSparse<0>offset_startpoint_array;

	uint64_t inputsizex64, my_block_length, my_batch_length;

	std::array<uint64_t, 21 * 21>combi_table;
	std::array<uint64_t, 21>combi_table_bitlength;

	void CalculateCombiTable() {

		uint64_t factorial[21];
		factorial[0] = 1;
		for (uint64_t i = 1; i <= 20; ++i)factorial[i] = i * factorial[i - 1];
		//ちなみに 20! < 2^64 < 21! で、 block_length <= 20 の制限はこれにも都合がいい

		for (uint64_t i = 0; i < 21 * 21; ++i)combi_table[i] = 0;

		//combination(i,j)
		for (uint64_t i = 1; i <= 20; ++i) {
			for (uint64_t j = 0; j <= i; ++j) {
				const uint64_t index = i * 21 + j;
				combi_table[index] = factorial[i] / (factorial[j] * factorial[i - j]);
			}
		}

		//combination((block_length ? block_length : my_block_length),i)のビット数
		for (int i = 0; i <= (block_length ? block_length : my_block_length); ++i) {
			combi_table_bitlength[i] = max(1ULL, log2_ceiling(combi_table[(block_length ? block_length : my_block_length) * 21 + i]));
		}
	}

	uint64_t Block2Offset(const uint64_t block) {
		uint64_t offset = 0;
		uint64_t pop = 1;
		for (uint64_t i = 0; i < (block_length ? block_length : my_block_length); ++i) {
			if (block & (1ULL << i)) {
				const uint64_t a = i;
				const uint64_t b = pop;
				offset += combi_table[a * 21 + b];
				++pop;
			}
		}
		return offset;
	}

	uint64_t Offset2Block(uint64_t offset, uint64_t pop) {
		uint64_t answer = 0;
		for (int64_t i = (block_length ? block_length : my_block_length) - 1; i >= 0 && pop; --i) {
			const uint64_t a = i;
			const uint64_t b = pop;
			const uint64_t c = combi_table[a * 21 + b];
			if (c <= offset) {
				offset -= c;
				answer |= 1ULL << i;
				--pop;
			}
		}
		return answer;
	}

public:

	BitVectorRRR(const std::vector<uint64_t>&input) {


		inputsizex64 = input.size() * 64;
		my_block_length = uint64_t(std::log2(input.size() * 64 / 2) + 0.5);
		my_batch_length = my_block_length * 2;

		CalculateCombiTable();

		const uint64_t block_num = (input.size() * 64 + (block_length ? block_length : my_block_length) - 1) / (block_length ? block_length : my_block_length);
		const uint64_t batch_num = (block_length ? block_length : my_block_length) * 2;
		const uint64_t global_num = (block_num + batch_num - 1) / batch_num;

		rank_global = std::vector<uint64_t>();
		offset_global = std::vector<uint64_t>();

		if (block_length) {
			block_class = NBitArray<block_length ? log2_ceiling_constexpr(block_length + 1) : 0>(block_num);
		}
		else {
			block_class = NBitArray<block_length ? log2_ceiling_constexpr(block_length + 1) : 0>(block_num, log2_ceiling(my_block_length + 1));
		}
		std::vector<uint64_t>tmp_block_offset_value;
		std::vector<uint64_t>tmp_block_offset_startpoint;
		std::vector<uint64_t>tmp_block_offset_bitlength(block_num);


		uint64_t offset_bit_sum = 0, rank_sum = 0;
		for (uint64_t i = 0; i < block_num; ++i) {

			if (i % batch_num == 0) {
				rank_global.push_back(rank_sum);
				offset_global.push_back(offset_bit_sum);
			}

			uint64_t block = 0;
			for (uint64_t j = 0; j < (block_length ? block_length : my_block_length); ++j) {
				const uint64_t bit = i * (block_length ? block_length : my_block_length) + j;
				if (bit >= input.size() * 64)break;
				if (input[bit / 64] & (1ULL << (bit % 64)))block |= 1ULL << j;
			}

			const uint64_t pop = popcount(block);
			const uint64_t offset = Block2Offset(block);

			rank_sum += pop;

			block_class.Set(i, pop);

			tmp_block_offset_bitlength[i] = combi_table_bitlength[pop];

			if (offset_bit_sum + combi_table_bitlength[pop] >= tmp_block_offset_value.size() * 64) {
				tmp_block_offset_value.push_back(0);
				tmp_block_offset_startpoint.push_back(0);
			}
			for (uint64_t j = 0; j < combi_table_bitlength[pop]; ++j) {
				if (offset & (1ULL << j)) {
					tmp_block_offset_value[offset_bit_sum / 64] |= 1ULL << (offset_bit_sum % 64);
				}
				if (j == 0) {
					tmp_block_offset_startpoint[offset_bit_sum / 64] |= 1ULL << (offset_bit_sum % 64);
				}
				++offset_bit_sum;
			}
		}
		rank_global.push_back(rank_sum);

		if (offset_bit_sum >= tmp_block_offset_value.size() * 64) {
			tmp_block_offset_startpoint.push_back(0);
		}
		tmp_block_offset_startpoint[offset_bit_sum / 64] |= 1ULL << (offset_bit_sum % 64);


		block_offset = BitVector<block_length ? median_constexpr(8, 16, log2_ceiling_constexpr((block_length * block_length * 4) * 2 / 3)) : 0>(tmp_block_offset_value);
		offset_startpoint_array = BitVectorSparse<0>(tmp_block_offset_startpoint);
	}

	BitVectorRRR() {
		BitVectorRRR(std::vector<uint64_t>{0});
	}

	bool access(const uint64_t index) {
		if (index >= inputsizex64)return false;

		const uint64_t block_index = index / (block_length ? block_length : my_block_length);
		const uint64_t pop = block_class.Get(block_index);

		const uint64_t pos0 = offset_startpoint_array.select1(block_index);
		const uint64_t pos1 = offset_startpoint_array.select1(block_index + 1);

		const uint64_t offset_length = pos1 - pos0;
		uint64_t offset = 0;
		for (uint64_t i = 0; i < offset_length; ++i) {
			if (block_offset.access(i + pos0))offset |= 1ULL << i;
		}
		const uint64_t block = Offset2Block(offset, pop);
		return (block & (1ULL << (index % (block_length ? block_length : my_block_length)))) ? true : false;
	}

	uint64_t rank1(const uint64_t index) {

		if (index >= inputsizex64)return rank_global.back();

		const uint64_t block_index = index / (block_length ? block_length : my_block_length);
		const uint64_t batch_index = block_index / (block_length ? (block_length * 2) : my_batch_length);

		uint64_t answer = rank_global[batch_index];
		for (uint64_t i = batch_index * (block_length ? (block_length * 2) : my_batch_length); i < block_index; ++i) {
			answer += block_class.Get(i);
		}

		const uint64_t pop = block_class.Get(block_index);
		const uint64_t pos0 = offset_startpoint_array.select1(block_index);
		const uint64_t pos1 = offset_startpoint_array.select1(block_index + 1);

		const uint64_t offset_length = pos1 - pos0;
		uint64_t offset = 0;
		for (uint64_t i = 0; i < offset_length; ++i) {
			if (block_offset.access(i + pos0))offset |= 1ULL << i;
		}
		const uint64_t block = Offset2Block(offset, pop);
		const uint64_t width = index - block_index * (block_length ? block_length : my_block_length);
		return answer + popcount(block & ((1ULL << width) - 1ULL));
	}
	uint64_t rank0(const uint64_t index) {
		if (index >= inputsizex64)return inputsizex64 - rank_global.back();
		return index - rank1(index);
	}

	uint64_t select1(const uint64_t count) {

		if (rank_global.back() <= count)return std::numeric_limits<uint64_t>::max();

		//global表の上で二分探索する。
		uint64_t lb = 0, ub = rank_global.size();
		while (lb + 1 < ub) {
			const uint64_t mid = (lb + ub) / 2;
			if (count < rank_global[mid])ub = mid;
			else lb = mid;
		}

		//この時点で、欲しい位置は
		//[lb*(block_length ? (block_length * 2) : my_batch_length),(lb+1)*(block_length ? (block_length * 2) : my_batch_length))
		//のどこかにある。

		//popを線形探索して、block内を線形探索する。

		uint64_t now_block_index = lb * (block_length ? (block_length * 2) : my_batch_length);
		uint64_t now_rank = rank_global[lb];
		for (;; now_block_index++) {
			const uint64_t pop = block_class.Get(now_block_index);
			if (now_rank + pop > count) {
				const uint64_t pos0 = offset_startpoint_array.select1(now_block_index);
				const uint64_t pos1 = offset_startpoint_array.select1(now_block_index + 1);

				const uint64_t offset_length = pos1 - pos0;
				uint64_t offset = 0;
				for (uint64_t i = 0; i < offset_length; ++i) {
					if (block_offset.access(i + pos0))offset |= 1ULL << i;
				}
				const uint64_t block = Offset2Block(offset, pop);

				//blockのビット列うち、
				//i番目のビットが立っていて、かつ[0,i)の中で(count-now_rank)個のビットが立っているようなiを求め、
				//(now_block_index*(block_length ? block_length : my_block_length)+i)を答えとして返す。

				//TODO: もっと速くてかっこいいビット演算があるはず……bsfとか使って
				for (uint64_t i = 0; i < 64; ++i) {
					if (block & (1ULL << i)) {
						if (count == now_rank++)return now_block_index * (block_length ? block_length : my_block_length) + i;
					}
				}

				assert(0);
				break;
			}
			now_rank += pop;
		}

		assert(0);
		return 0;
	}

	uint64_t select0(const uint64_t count) {

		if (inputsizex64 - rank_global.back() <= count)return std::numeric_limits<uint64_t>::max();

		//global表の上で二分探索する。
		uint64_t lb = 0, ub = rank_global.size();
		while (lb + 1 < ub) {
			const uint64_t mid = (lb + ub) / 2;
			if (count < mid * (block_length ? block_length : my_block_length) * (block_length ? (block_length * 2) : my_batch_length) - rank_global[mid])ub = mid;
			else lb = mid;
		}

		uint64_t now_block_index = lb * (block_length ? (block_length * 2) : my_batch_length);
		uint64_t now_rank = lb * (block_length ? block_length : my_block_length) * (block_length ? (block_length * 2) : my_batch_length) - rank_global[lb];
		for (;; now_block_index++) {
			const uint64_t pop = block_class.Get(now_block_index);
			if (now_rank + (block_length ? block_length : my_block_length) - pop > count) {
				const uint64_t pos0 = offset_startpoint_array.select1(now_block_index);
				const uint64_t pos1 = offset_startpoint_array.select1(now_block_index + 1);

				const uint64_t offset_length = pos1 - pos0;
				uint64_t offset = 0;
				for (uint64_t i = 0; i < offset_length; ++i) {
					if (block_offset.access(i + pos0))offset |= 1ULL << i;
				}
				const uint64_t block = Offset2Block(offset, pop);

				//blockのビット列うち、
				//i番目のビットが立っていなくて、かつ[0,i)の中で(count-now_rank)個のビットが立っていないようなiを求め、
				//(now_block_index+(block_length ? block_length : my_block_length)+i)を答えとして返す。

				//TODO: もっと速くてかっこいいビット演算があるはず……bsfとか使って
				for (uint64_t i = 0; i < 64; ++i) {
					if (!(block & (1ULL << i))) {
						if (count == now_rank++)return now_block_index * (block_length ? block_length : my_block_length) + i;
					}
				}

				assert(0);
				break;
			}
			now_rank += (block_length ? block_length : my_block_length) - pop;
		}

		assert(0);
		return 0;
	}

	uint64_t sum1() {
		return rank_global.back();
	}

};

//SuffixArrayを圧縮して持っておいて、参照クエリに答える。
template<uint64_t compress_order>class CompressedSuffixArray {

	//2^compress_order個ごとにサンプルする。

private:

	VerticalCodePermutation<uint8_t> psi_vertical_code;
	std::vector<uint64_t>suffix_array_sumple;
	BitVectorSparse<compress_order> B;
	int64_t suffix_array_size;

public:

	CompressedSuffixArray(const std::vector<uint8_t>& input) {

		assert(1 <= compress_order && compress_order <= 10);
		for (int64_t i = 0; i < input.size() - 1; ++i)assert(input[i] != 0);
		assert(input.back() == 0);

		SuffixArrayMaker sa;
		const auto suffix_array = sa.SuffixArrayConstruction(input);
		suffix_array_size = suffix_array.size();

		//Inverse Suffix Arrayを作る。
		std::vector<int>inverse_suffix_array(suffix_array.size(), 0);
		for (int64_t i = 0; i < inverse_suffix_array.size(); ++i) {
			inverse_suffix_array[suffix_array[i]] = i;
		}

		//Inverse Suffix Arrayを使ってPsi Arrayを作り、Vertical Codeにする。
		std::vector<uint64_t>psi(suffix_array.size(), 0);
		for (int64_t i = 0; i < psi.size(); ++i) {
			psi[i] =
				(suffix_array[i] == suffix_array.size() - 1) ?
				inverse_suffix_array[0] :
				inverse_suffix_array[suffix_array[i] + 1];
		}
		psi_vertical_code = VerticalCodePermutation<uint8_t>(psi);

		//ビット列Bを作る。定義は
		//Bのi番目のビットがtrue⇔SA[i]が2^compress_orderで割り切れる
		//とする。(元の岡野原論文の定義)
		//加えてSA[i]==SA.size()-1のときもサンプルを保存しておく。
		//そうすることでGetのときに剰余を取る必要がなくなる。
		const auto interval = uint64_t(1) << compress_order;
		std::vector<uint64_t>b_tmp((suffix_array.size() + 63) / 64);
		for (int i = 0; i < suffix_array.size(); ++i) {
			if ((suffix_array[i] % interval) == 0 || suffix_array[i] == suffix_array.size() - 1) {
				b_tmp[i / 64] |= uint64_t(1) << (i % 64);
			}
		}
		B = BitVectorSparse<compress_order>(b_tmp);

		//Bのビットが立っている位置のsuffix arrayだけをサンプルして保存しておく。
		const int64_t sample_size = B.sum1();
		suffix_array_sumple = std::vector<uint64_t>(sample_size, 0);
		for (int64_t i = 0; i < sample_size; ++i) {
			suffix_array_sumple[i] = suffix_array[B.select1(i)];
		}
	}

	uint64_t Get(int64_t index) {

		//suffix_array[index]を返す。

		for (int64_t move_count = 0;; ++move_count) {

			if (B.access(index)) {

				//剰余について
				//この記述の中ではk=2^compress_order, t=move_countとする。
				//元の岡野原論文の定義だと、B.access(index)がtrueになる⇔SA[index]がkで割り切れる　であった。
				//B.access(index)がfalseの場合、index:=psi[index]とすることで、"返り値が1増える"効果があるので、
				//たかだかk-1回繰り返しpsiを作用させることで、B.access(index)がtrueになる。
				//そのとき、psiを作用させた回数をtとすると、SA_sample[B.rank(index)]-tを返せば良いと書かれていた。
				//しかし実際には、SA.size()がkで割り切れないとき、SA_sample[B.rank(index)]==0になることがある。
				//そのため、元論文の通りSA[index]がkで割り切れるもののみサンプリングしたなら、
				//元論文と違いSA.size()で割った余りを返す必要がある。
				//今回の実装では元論文と違いSA[index]==SA.size()-1のときもサンプリングしているので、
				//tが非ゼロのときSA_sample[B.rank(index)]==0にはならず、剰余の処理は必要ない。

				return (int64_t(suffix_array_sumple[B.rank1(index)]) - move_count);
				//return (int64_t(suffix_array_sumple[B.rank(index)]) - move_count + suffix_array_size) % suffix_array_size;
			}
			index = psi_vertical_code.Get(index);
		}

		assert(0);
		return 0;
	}
};

//文字列を受け取ってBWTして返す。内部でSuffix Arrayの構築をしているのは特に意味なくて、実運用のときに省略できればするとよい。
std::vector<uint8_t>BWT(const std::vector<uint8_t>& input) {

	for (int i = 0; i < input.size() - 1; ++i)assert(input[i] != 0);
	assert(input.back() == 0);

	SuffixArrayMaker sa;
	const auto suffix_array = sa.SuffixArrayConstruction(input);

	std::vector<uint8_t>bwt(suffix_array.size(), 0);
	for (int i = 0; i < suffix_array.size(); ++i) {
		if (suffix_array[i] == 0)bwt[i] = input.back();
		else bwt[i] = input[suffix_array[i] - 1];
	}
	return bwt;
}

//文字列を持っておいて、Rankクエリに時間O(logC)で答える。Cはアルファベットの種類数。
class WaveletMatrix {
private:

	std::vector<BitVectorRRR<16>>bv;
	std::vector<int64_t>acc;
	int64_t size;

	static uint64_t popcount(uint64_t x) {
		x = (x & 0x5555555555555555ULL) + ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
		x = (x & 0x3333333333333333ULL) + ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
		x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
		x *= 0x0101010101010101ULL;
		return x >> 56;
	}

	//x以上の2べき乗数のうち最小の数が2^kとしてkを返す。
	static int64_t log2_ceiling(uint64_t x) {

		if (x <= 0)return 0;
		x--;
		//msbの位置を求めるために、ビットを下位に伝播させてpopcntする。
		for (int i = 0; i < 6; ++i)x |= x >> (1ULL << i);
		return popcount(x);
	}

public:

	WaveletMatrix(const std::vector<uint8_t>& input, const int max_char) {
		//inputは入力文字列で、max_charは入力文字列が含みうる文字の最大値。
		//実際に含んでいる文字の最大値ではない。これを用意する理由は高速化で、
		//具体的にはrank操作においてmax_charを超える引数を受け取らないと仮定する。

		assert(1 <= max_char && max_char <= 255);
		for (const auto x : input)assert(x <= max_char);

		size = input.size();

		//max_charのビット数ぶんBitVectorを用意する。
		const int64_t bits = log2_ceiling(max_char + 1);
		bv = std::vector<BitVectorRRR<16>>(bits, BitVectorRRR<16>());

		//BitVectorの最上位は、inputの最上位ビットの列を格納する。
		std::vector<uint64_t>bv_tmp((input.size() + 63) / 64, 0);
		const auto SetBit1 = [&](const int i) {bv_tmp[i / 64] |= uint64_t(1) << (i % 64); };
		for (int i = 0; i < input.size(); ++i) {
			if (input[i] & (1 << (bits - 1)))SetBit1(i);
		}
		bv[bits - 1] = BitVectorRRR<16>(bv_tmp);

		auto input_tmp = input;
		//BitVectorの最上位以外は、input_tmp配列を(bit_pos+1)番目のビットで安定ソートした後における、
		//bit_pos番目のビットの列を格納する。cf.高速文字列解析の世界p74
		for (int bit_pos = bits - 2; bit_pos >= 0; --bit_pos) {
			int count = 0;
			std::vector<uint8_t>input_tmp2(input.size(), 0);
			for (int i = 0; i < input.size(); ++i) {
				if ((input_tmp[i] & (1 << (bit_pos + 1))) == 0)input_tmp2[count++] = input_tmp[i];
			}
			for (int i = 0; i < input.size(); ++i) {
				if ((input_tmp[i] & (1 << (bit_pos + 1))) != 0)input_tmp2[count++] = input_tmp[i];
			}
			for (int i = 0; i < bv_tmp.size(); ++i)bv_tmp[i] = 0;
			for (int i = 0; i < input.size(); ++i) {
				if (input_tmp2[i] & (1 << bit_pos))SetBit1(i);
			}
			bv[bit_pos] = BitVectorRRR<16>(bv_tmp);
			input_tmp = input_tmp2;
		}

		{
			int count = 0;
			std::vector<uint8_t>input_tmp2(input.size(), 0);
			for (int i = 0; i < input.size(); ++i) {
				if ((input_tmp[i] & 1) == 0)input_tmp2[count++] = input_tmp[i];
			}
			for (int i = 0; i < input.size(); ++i) {
				if ((input_tmp[i] & 1) != 0)input_tmp2[count++] = input_tmp[i];
			}
			//acc[i]にはinput_tmp2中でiが最初に出現した位置を格納する。出現しない場合は-1とする。
			acc = std::vector<int64_t>(max_char + 1, -1);
			for (int i = 0; i < input_tmp2.size(); ++i)if (acc[input_tmp2[i]] == -1)acc[input_tmp2[i]] = i;
		}
	}

	int64_t rank(const int64_t index, const uint8_t character) {
		//リファレンス文字列の[0,index)のうち、characterが何回出現したか返す。

		if (acc[character] == -1)return 0;

		//リファレンス文字列の[0,index)のうち、最上位ビットがcharacterに等しいものを数える。
		uint64_t tmp_rank =
			(character & (1 << (bv.size() - 1))) ?
			bv[bv.size() - 1].rank1(index) :
			bv[bv.size() - 1].rank0(index);

		//最上位-1ビットから下位ビットに向けて処理していく。
		for (int bit_pos = bv.size() - 2; bit_pos >= 0; --bit_pos) {
			if (character & (1 << (bit_pos + 1))) {
				const uint64_t start_pos = size - bv[bit_pos + 1].sum1();
				tmp_rank = (character & (1 << bit_pos)) ?
					bv[bit_pos].rank1(start_pos + tmp_rank) :
					bv[bit_pos].rank0(start_pos + tmp_rank);
			}
			else {
				tmp_rank = (character & (1 << bit_pos)) ?
					bv[bit_pos].rank1(tmp_rank) :
					bv[bit_pos].rank0(tmp_rank);
			}
		}
		if (character & 1)tmp_rank += size - bv[0].sum1();

		return tmp_rank - acc[character];
	}
};

//TODO: 機能はするはずだが、ライブラリとして適した形になってないのでする
std::vector<int> FMIndexMatch(
	const std::vector<uint8_t>& ref,
	const std::vector<uint8_t>& query) {

	for (const auto x : ref)assert(x != 0);
	for (const auto x : query)assert(x != 0);

	auto ref_ = ref;
	ref_.push_back(0);
	SuffixArrayMaker sa;
	const auto suffix_array = sa.SuffixArrayConstruction(ref_);
	const auto bwt = BWT(ref_);
	//BruteForceWaveletMatrix wavelet_matrix(bwt);
	WaveletMatrix wavelet_matrix(bwt, 255);

	//bwt中でのiより小さい文字の出現回数を求めてacc_num[i]に格納する。
	std::vector<int64_t>num(256, 0), acc_num(256, 0);
	for (const auto x : bwt)++num[x];
	for (int i = 1; i < 256; ++i)acc_num[i] = num[i - 1] + acc_num[i - 1];

	int64_t start_pos = 0, end_pos = ref_.size();
	for (int i = query.size() - 1; i >= 0; --i) {
		const auto c = query[i];
		start_pos = acc_num[c] + wavelet_matrix.rank(start_pos, c);
		end_pos = acc_num[c] + wavelet_matrix.rank(end_pos, c);
		if (start_pos >= end_pos)return std::vector<int>{};
	}

	std::vector<int>match_start_pos;
	for (int64_t i = start_pos; i < end_pos; ++i)match_start_pos.push_back(suffix_array[i]);
	sort(match_start_pos.begin(), match_start_pos.end());
	return match_start_pos;
}



std::vector<int> BruteForceSuffixArrayConstruction(const std::vector<uint8_t>& input) {

	//終端文字は0で、終端文字以外は正の数だと仮定する。
	for (int i = 0; i < input.size() - 1; ++i)assert(input[i] > 0);
	assert(input.back() == 0);

	//suffixを全列挙してソートする。
	std::vector<std::vector<uint8_t>>suffix;
	for (int i = 0; i < input.size(); ++i) {
		std::vector<uint8_t> s;
		for (int j = i; j < input.size(); ++j)s.push_back(input[j]);
		suffix.push_back(s);
	}
	sort(suffix.begin(), suffix.end());

	//ソートしたときの順番を記録して返す。ここでは順番をsuffixの長さから求めている。
	std::vector<int>suffix_array;
	for (int i = 0; i < suffix.size(); ++i) {
		suffix_array.push_back(input.size() - suffix[i].size());
	}
	return suffix_array;
}
std::vector<int>BruteForceLCPArrayConstruction(
	const std::vector<uint8_t>& input,
	const std::vector<int>& suffix_array) {

	for (int i = 0; i < input.size() - 1; ++i)assert(input[i] != 0);
	assert(input.back() == 0);
	assert(input.size() == suffix_array.size());

	std::vector<int>lcp_array(input.size() - 1, 0);
	for (int i = 0; i < input.size() - 1; ++i) {
		while (
			input[suffix_array[i] + lcp_array[i]] ==
			input[suffix_array[i + 1] + lcp_array[i]]) {
			lcp_array[i]++;
		}
	}

	return lcp_array;
}
class BruteForceBitVector {
private:

	std::vector<uint64_t>B;

public:

	BruteForceBitVector(const std::vector<uint64_t>& input) {
		B = input;
	}
	BruteForceBitVector() {
		BruteForceBitVector(std::vector<uint64_t>{0});
	}

	bool access(const uint64_t index) {
		return (B[index / 64] & (uint64_t(1) << (index % 64))) != 0;
	}

	uint64_t rank1(const uint64_t index) {
		//[0,index)にいくつ立っているビットがあるか求めて返す。
		uint64_t answer = 0;
		for (int i = 0; i < index; ++i)if (access(i))++answer;
		return answer;
	}
	uint64_t rank0(const uint64_t index) {
		//[0,index)にいくつ立っていないビットがあるか求めて返す。
		uint64_t answer = 0;
		for (int i = 0; i < index; ++i)if (!access(i))++answer;
		return answer;
	}

	uint64_t select1(const uint64_t count) {
		//count番目(0-origin)の1のビットの位置を求めて返す。
		uint64_t acc = 0;
		for (int i = 0; i < B.size() * 64; ++i)if (access(i))if (acc++ == count)return i;
		return std::numeric_limits<uint64_t>::max();
	}
	uint64_t select0(const uint64_t count) {
		//count番目(0-origin)の0のビットの位置を求めて返す。
		uint64_t acc = 0;
		for (int i = 0; i < B.size() * 64; ++i)if (!access(i))if (acc++ == count)return i;
		return std::numeric_limits<uint64_t>::max();
	}
};
class BruteForceWaveletMatrix {
private:

	std::vector<uint8_t>v;

public:

	BruteForceWaveletMatrix(const std::vector<uint8_t>& input) {
		v = input;
	}

	int64_t rank(const int64_t index, const uint8_t character) {
		//リファレンス文字列の[0,index)のうち、characterが何回出現したか返す。
		int64_t answer = 0;
		for (int i = 0; i < index; ++i)if (v[i] == character)++answer;
		return answer;
	}
};

void TestVerticalCode(const int num) {

	std::mt19937_64 rnd(num);
	std::uniform_int_distribution<int>rand_value(0, 1 + (num % 9));

	const int length = num % 1000 + 100;
	std::vector<uint64_t>arr(length, 0);
	for (int i = 0; i < length; ++i)arr[i] = i;
	std::shuffle(arr.begin(), arr.end(), rnd);

	{
		VerticalCodePermutation<uint64_t> vc(arr);
		for (int i = 0; i < length; ++i)assert(vc.Get(i) == arr[i]);
	}
	{
		VerticalCodePermutation<uint32_t> vc(arr);
		for (int i = 0; i < length; ++i)assert(vc.Get(i) == arr[i]);
	}
	{
		VerticalCodePermutation<uint16_t> vc(arr);
		for (int i = 0; i < length; ++i)assert(vc.Get(i) == arr[i]);
	}
	{
		VerticalCodePermutation<uint8_t> vc(arr);
		for (int i = 0; i < length; ++i)assert(vc.Get(i) == arr[i]);
	}

	std::vector<uint64_t>sum(length, 0);
	sum[0] = rand_value(rnd);
	for (int i = 1; i < length; ++i)sum[i] = sum[i - 1] + rand_value(rnd);

	{
		VerticalCodeWeaklyIncrease<uint64_t> vc(sum);
		for (int i = 0; i < length; ++i)assert(vc.Get(i) == sum[i]);
	}
	{
		VerticalCodeWeaklyIncrease<uint32_t> vc(sum);
		for (int i = 0; i < length; ++i)assert(vc.Get(i) == sum[i]);
	}
	{
		VerticalCodeWeaklyIncrease<uint16_t> vc(sum);
		for (int i = 0; i < length; ++i)assert(vc.Get(i) == sum[i]);
	}
	{
		VerticalCodeWeaklyIncrease<uint8_t> vc(sum);
		for (int i = 0; i < length; ++i)assert(vc.Get(i) == sum[i]);
	}



}
void TestBitVector(const int num) {

	std::mt19937_64 rnd(num);

	const int length = 100 + num % 1000;
	std::vector<uint64_t>arr(length, 0);
	for (int i = 0; i < length; ++i)arr[i] = (num % 2) ? rnd() : (rnd()&rnd()&rnd()&rnd());
	BruteForceBitVector bv1(arr);
	BitVector<10> bv2(arr);
	BitVector<0> bv3(arr);
	BitVectorSparse<3> bv4(arr);
	BitVectorSparse<0> bv5(arr);
	BitVectorRRR<6> bv6(arr);
	BitVectorRRR<0> bv7(arr);
	int pop = 0;
	for (int i = 0; i < length; ++i) pop += popcount(arr[i]);
	assert(bv2.sum1() == pop);
	for (int i = 0; i < length * 64; ++i) {
		const bool b1 = bv1.access(i);
		const bool b2 = bv2.access(i);
		const bool b3 = bv3.access(i);
		const bool b4 = bv4.access(i);
		const bool b5 = bv5.access(i);
		const bool b6 = bv6.access(i);
		const bool b7 = bv7.access(i);
		if (b1 != b2) {
			bv1.access(i);
			bv2.access(i);
			assert(0);
		}
		if (b1 != b3) {
			bv1.access(i);
			bv3.access(i);
			assert(0);
		}
		if (b1 != b4) {
			bv1.access(i);
			bv4.access(i);
			assert(0);
		}
		if (b1 != b5) {
			bv1.access(i);
			bv5.access(i);
			assert(0);
		}
		if (b1 != b6) {
			bv1.access(i);
			bv6.access(i);
			assert(0);
		}
		if (b1 != b7) {
			bv1.access(i);
			bv7.access(i);
			assert(0);
		}
		const uint64_t r11 = bv1.rank1(i);
		const uint64_t r12 = bv2.rank1(i);
		const uint64_t r13 = bv3.rank1(i);
		const uint64_t r14 = bv4.rank1(i);
		const uint64_t r15 = bv5.rank1(i);
		const uint64_t r16 = bv6.rank1(i);
		const uint64_t r17 = bv7.rank1(i);
		if (r11 != r12) {
			bv1.rank1(i);
			bv2.rank1(i);
			assert(0);
		}
		if (r11 != r13) {
			bv1.rank1(i);
			bv3.rank1(i);
			assert(0);
		}
		if (r11 != r14) {
			bv1.rank1(i);
			bv4.rank1(i);
			assert(0);
		}
		if (r11 != r15) {
			bv1.rank1(i);
			bv5.rank1(i);
			assert(0);
		}
		if (r11 != r16) {
			bv1.rank1(i);
			bv6.rank1(i);
			assert(0);
		}
		if (r11 != r17) {
			bv1.rank1(i);
			bv7.rank1(i);
			assert(0);
		}
		const uint64_t r01 = bv1.rank0(i);
		const uint64_t r02 = bv2.rank0(i);
		const uint64_t r03 = bv3.rank0(i);
		const uint64_t r04 = bv4.rank0(i);
		const uint64_t r05 = bv5.rank0(i);
		const uint64_t r06 = bv6.rank0(i);
		const uint64_t r07 = bv7.rank0(i);
		if (r01 != r02) {
			bv1.rank0(i);
			bv2.rank0(i);
			assert(0);
		}
		if (r01 != r03) {
			bv1.rank0(i);
			bv3.rank0(i);
			assert(0);
		}
		if (r01 != r04) {
			bv1.rank0(i);
			bv4.rank0(i);
			assert(0);
		}
		if (r01 != r05) {
			bv1.rank0(i);
			bv5.rank0(i);
			assert(0);
		}
		if (r01 != r06) {
			bv1.rank0(i);
			bv6.rank0(i);
			assert(0);
		}
		if (r01 != r07) {
			bv1.rank0(i);
			bv7.rank0(i);
			assert(0);
		}
	}
	for (int i = 0; i <= pop + 1; ++i) {
		const uint64_t s11 = bv1.select1(i);
		const uint64_t s12 = bv2.select1(i);
		const uint64_t s13 = bv3.select1(i);
		const uint64_t s14 = bv4.select1(i);
		const uint64_t s15 = bv5.select1(i);
		const uint64_t s16 = bv6.select1(i);
		const uint64_t s17 = bv7.select1(i);
		if (s11 != s12) {
			bv1.select1(i);
			bv2.select1(i);
			assert(0);
		}
		if (s11 != s13) {
			bv1.select1(i);
			bv3.select1(i);
			assert(0);
		}
		if (s11 != s14) {
			bv1.select1(i);
			bv4.select1(i);
			assert(0);
		}
		if (s11 != s15) {
			bv1.select1(i);
			bv5.select1(i);
			assert(0);
		}
		if (s11 != s16) {
			bv1.select1(i);
			bv6.select1(i);
			assert(0);
		}
		if (s11 != s17) {
			bv1.select1(i);
			bv7.select1(i);
			assert(0);
		}
		const uint64_t s01 = bv1.select0(i);
		const uint64_t s02 = bv2.select0(i);
		const uint64_t s03 = bv3.select0(i);
		const uint64_t s04 = bv4.select0(i);
		const uint64_t s05 = bv5.select0(i);
		const uint64_t s06 = bv6.select0(i);
		const uint64_t s07 = bv7.select0(i);
		if (s01 != s02) {
			bv1.select0(i);
			bv2.select0(i);
			assert(0);
		}
		if (s01 != s03) {
			bv1.select0(i);
			bv3.select0(i);
			assert(0);
		}
		if (s01 != s04) {
			bv1.select0(i);
			bv4.select0(i);
			assert(0);
		}
		if (s01 != s05) {
			bv1.select0(i);
			bv5.select0(i);
			assert(0);
		}
		if (s01 != s06) {
			bv1.select0(i);
			bv6.select0(i);
			assert(0);
		}
		if (s01 != s07) {
			bv1.select0(i);
			bv7.select0(i);
			assert(0);
		}
	}
}
void TestSA(const int num) {
	std::mt19937_64 rnd(num);
	std::uniform_int_distribution<int>rand_value(1, 4);

	const int ref_len1 = num % 1000 + 100;

	std::vector<uint8_t>ref_text1(ref_len1, 0);
	for (int i = 0; i < ref_len1; ++i)ref_text1[i] = uint8_t(rand_value(rnd));
	ref_text1.back() = 0;

	SuffixArrayMaker sa;
	const auto sa1 = BruteForceSuffixArrayConstruction(ref_text1);
	const auto sa2 = sa.SuffixArrayConstruction(ref_text1);
	const auto lcp1 = BruteForceLCPArrayConstruction(ref_text1, sa1);
	const auto lcp2 = sa.KasaiLCPArrayConstruction(ref_text1, sa2);

	assert(sa1.size() == sa2.size());
	assert(lcp1.size() == lcp2.size());
	assert(sa1.size() == lcp1.size() + 1);

	for (int i = 0; i < sa1.size(); ++i) {
		assert(sa1[i] == sa2[i]);
		if (i)assert(lcp1[i - 1] == lcp2[i - 1]);
	}
	{
		CompressedSuffixArray<3>sa3(ref_text1);
		for (int i = 0; i < sa1.size(); ++i) {
			assert(sa3.Get(i) == sa1[i]);
		}
	}
	{
		CompressedSuffixArray<4>sa3(ref_text1);
		for (int i = 0; i < sa1.size(); ++i) {
			assert(sa3.Get(i) == sa1[i]);
		}
	}
	{
		CompressedSuffixArray<5>sa3(ref_text1);
		for (int i = 0; i < sa1.size(); ++i) {
			assert(sa3.Get(i) == sa1[i]);
		}
	}
	{
		CompressedSuffixArray<6>sa3(ref_text1);
		for (int i = 0; i < sa1.size(); ++i) {
			assert(sa3.Get(i) == sa1[i]);
		}
	}
}
void TestWaveletMatrix(const int num) {
	std::mt19937_64 rnd(num);

	for (int bbb = 1; bbb <= 255; ++bbb) {

		std::uniform_int_distribution<int>rand_value(0, bbb);
		const int ref_len1 = num % 1000 + 100;

		std::vector<uint8_t>ref_text1(ref_len1, 0);
		for (int i = 0; i < ref_len1; ++i)ref_text1[i] = uint8_t(rand_value(rnd));

		BruteForceWaveletMatrix m1(ref_text1);
		WaveletMatrix m2(ref_text1, bbb);

		for (int i = 0; i < ref_len1; ++i) {
			for (int ccc = 0; ccc <= bbb; ++ccc) {
				const auto r1 = m1.rank(i, ccc);
				const auto r2 = m2.rank(i, ccc);
				assert(r1 == r2);
			}
		}
	}
}


signed main() {

//#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < 10000; ++i) {
//#pragma omp critical
		{
			cout << i << endl;
		}
		TestWaveletMatrix(i);
		TestBitVector(i);
		TestSA(i);
		TestVerticalCode(i);
	}

	return 0;
}