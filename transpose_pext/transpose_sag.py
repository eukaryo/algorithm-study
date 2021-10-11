
def pdep(a, mask):
    dest = 0
    k = 0
    for m in range(64):
        if (mask & (1 << m)) != 0:
            if (a & (1 << k)) != 0:
                dest += 1 << m
            k += 1
    return dest

def pext(a, mask):
    dest = 0
    k = 0
    for m in range(64):
        if (mask & (1 << m)) != 0:
            if (a & (1 << m)) != 0:
                dest += (1 << k)
            k += 1
    return dest

def popcount(n):
    return bin(n).count("1")

def compress(x, mask):
    return pext(x, mask)

def sag(x, mask):
    return (compress(x, mask) << (popcount(((1 << 64) - 1) ^ mask))) | compress(x, ((1 << 64) - 1) ^ mask)

def ensure_permutation_64(x):
    assert type(x) is list
    assert len(x) == 64
    y = sorted(x)
    assert sum([abs(y[i] - i) for i in list(range(64))]) == 0

def func(x):

    ensure_permutation_64(x)

    x_stable = [-1 for _ in range(64)]

    rep = 0
    count = 0
    while count < 64:
        for i in range(64):
            if x[i] == count:
                x_stable[i] = rep
                count += 1
        rep += 1

    assert min(x_stable) == 0
    return x_stable

def init_p_array(x):

    ensure_permutation_64(x)

    x_stable = func(x)

    p = [0 for _ in range(6)]

    # 2進6桁の値64個からなる配列xを「ビットごとに行列転置」して、
    # 2進64桁の値6個の配列pに格納する。
    for i in range(64):
        b = x_stable[i]
        j = 0
        while b != 0:
            if b % 2 == 1:
                p[j] |= 1 << i
            j += 1
            b >>= 1

    # ハッカーのたのしみ133ページの事前計算
    p[1] = sag(p[1], p[0])
    p[2] = sag(sag(p[2], p[0]), p[1])
    p[3] = sag(sag(sag(p[3], p[0]), p[1]), p[2])
    p[4] = sag(sag(sag(sag(p[4], p[0]), p[1]), p[2]), p[3])
    p[5] = sag(sag(sag(sag(sag(p[5], p[0]), p[1]), p[2]), p[3]), p[4])

    return p

if __name__ == "__main__":

    transpose = [-1 for _ in range(64)]
    for i in range(8):
        for j in range(8):
            transpose[i * 8 + j] = j * 8 + i
    for i in range(8):
        print(transpose[i*8:(i+1)*8])
    print(transpose)

    p = init_p_array(transpose)
    print([hex(x) for x in p])
    print([popcount(x) for x in p])