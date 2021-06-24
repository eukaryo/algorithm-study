#include<iostream>
#include<vector>

class UnionFind {

private:

	std::vector<int64_t> v;
	std::vector<int64_t> size;
	int64_t tree_num;

	int64_t find(int64_t a) {
		if (v[a] == a)return a;
		return v[a] = find(v[a]);
	}

public:

	UnionFind(const int64_t N) :v(N), size(N) {
		tree_num = N;
		for (int64_t i = 0; i < N; ++i) {
			v[i] = i;
			size[i] = 1;
		}
	}
	
	void make_set() {
		const int64_t N = v.size();
		v.push_back(N);
		size[N] = 1;
		tree_num++;
	}


	void unite(int64_t a, int64_t b) {
		a = find(a);
		b = find(b);
		if (a == b)return;

		if (size[a] < size[b])std::swap(a, b);
		v[b] = a;
		size[a] += size[b];
		tree_num--;
	}

	bool is_same(int64_t a, int64_t b) { return find(a) == find(b); }

	int get_size(int64_t a) {
		return size[find(a)];
	}
	int get_tree_num() { return tree_num; }
};

int main() {

	UnionFind uf(100);
	std::cout << (uf.is_same(24, 57) ? "same" : "not same") << std::endl;
	uf.unite(24, 57);
	std::cout << (uf.is_same(24, 57) ? "same" : "not same") << std::endl;
	std::cout << "size = " << uf.get_size(24) << std::endl;

	return 0;
}
