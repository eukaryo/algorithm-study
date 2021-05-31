import sys
import re
import hashlib

import pulp

import z3
# z3をimportする簡単な方法：
# https://qiita.com/SatoshiTerasaki/items/476c9938479a4bfdda52


def is_varid_sudoku_problem_string_format(s: str):
    if re.fullmatch(r"[1-9.]{81}", s) is None:
        return False
    return True


def print_sudoku(s: str):
    print("+---+---+---+")
    for i in range(0,81,27):
        for j in range(0,27,9):
            line = "|"
            for k in range(0,9,3):
                line = line + s[i+j+k:i+j+k+3] + "|"
            print(line)
        print("+---+---+---+")


def print_sikiri():
    print("")
    print("------------------------------------------------------------------------------------------")
    print("")


def flatten_list(l):
    answer = []
    for element in l:
        if type(element) is list:
            answer.extend(flatten_list(element))
        else:
            answer.append(element)
    return answer


def solve_pulp(sudoku_problem: str, known_answer=None):
    # https://coin-or.github.io/pulp/index.html
    # https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html
    # 53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79

    #LpProblem関数はデフォルトだと最小化問題を解く。今回は、充足可能なら必ず81になるので最小化最大化のどちらでもいい。
    lp_problem = pulp.LpProblem() #(sense=pulp.LpMaximize)

    # ソルバーに与える0-1変数を用意する。9*9*9個であり、特定の座標に特定の数字が入るなら1でさもなくば0とする。
    vals = pulp.LpVariable.dicts("Choice", (range(9), range(9), range(9)), cat="Binary")

    # 目的関数は「入れられる数字の数」とする。問題が解ければ81になり、解けなければinfeasibleとなる。
    # これを与えなくても充足可能かどうかは判定してくれるので与えなくても構わない。公式サイトの例では与えていない。
    lp_problem.setObjective(pulp.lpSum(flatten_list(vals)))

    # 各々のマスにはちょうど一つの数字が入るという制約。
    for y in range(9):
        for x in range(9):
            lp_problem += pulp.lpSum([vals[i][y][x] for i in range(9)]) == 1
    
    # 各々の行には各々の数字がちょうど一つだけ入るという制約。
    for i in range(9):
        for x in range(9):
            lp_problem += pulp.lpSum([vals[i][y][x] for y in range(9)]) == 1

    # 各々の列には各々の数字がちょうど一つだけ入るという制約。
    for i in range(9):
        for y in range(9):
            lp_problem += pulp.lpSum([vals[i][y][x] for x in range(9)]) == 1

    # 各々の枠（3*3に区切られた領域）には各々の数字がちょうど一つだけ入るという制約。
    for i in range(9):
        for y in range(0,9,3):
            for x in range(0,9,3):
                lp_problem += pulp.lpSum([vals[i][y+j//3][x+j%3] for j in range(9)]) == 1

    # 問題文で与えられたヒント（最初から埋まっている数字）も制約として与える。
    for y in range(9):
        for x in range(9):
            if sudoku_problem[y*9+x] != ".":
                n = int(sudoku_problem[y*9+x])
                lp_problem += vals[n-1][y][x] == 1

    # 第2引数で既知の解が与えられている場合、それ以外の解を探索したい。
    # そのため、「既知の解と完全一致してはいけない」という制約を与える。
    if known_answer is not None:
        if re.fullmatch(r"[1-9]{81}", known_answer) is None:
            raise ValueError("error: known_answer is invalid format.")
        a = [int(known_answer[i]) for i in range(81)]
        lp_problem += pulp.lpSum(flatten_list([[vals[a[y*9+x]-1][y][x] for y in range(9)] for x in range(9)])) <= 80

    lp_problem.solve(pulp.PULP_CBC_CMD(msg=0))

    #print("Status:", pulp.LpStatus[lp_problem.status])

    sudoku_answer_list = list(sudoku_problem)

    for i in range(9):
        for y in range(9):
            for x in range(9):
                if pulp.value(vals[i][y][x]) == 1:
                    sudoku_answer_list[y*9+x] = str(i+1)
    
    return "".join(sudoku_answer_list)


def solve_z3py(sudoku_problem: str, known_answer=None):
    # http://ericpony.github.io/z3py-tutorial/guide-examples.htm

    s = z3.Solver()
    
    # ソルバーに与える変数を用意する。9*9個の整数であり、特定の座標に入る数字を意味する。
    vals = [[z3.Int(f"val({y},{x})") for y in range(9)] for x in range(9)]

    # 入る数字は1以上9以下であるという制約。
    s.add([z3.And(1 <= vals[y][x], vals[y][x] <= 9) for y in range(9) for x in range(9)])

    # 各々の行には各々の数字がちょうど一つだけ入るという制約。
    s.add([z3.Distinct(vals[y]) for y in range(9)])

    # 各々の列には各々の数字がちょうど一つだけ入るという制約。
    s.add([z3.Distinct([vals[y][x] for y in range(9)]) for x in range(9)])

    # 各々の枠（3*3に区切られた領域）には各々の数字がちょうど一つだけ入るという制約。
    s.add([z3.Distinct([vals[y+i][x+j] for i in range(3) for j in range(3)]) for y in range(0,9,3) for x in range(0,9,3)])

    # 問題文で与えられたヒント（最初から埋まっている数字）も制約として与える。
    for y in range(9):
        for x in range(9):
            if sudoku_problem[y*9+x] != ".":
                n = int(sudoku_problem[y*9+x])
                s.add(vals[y][x] == n)

    # 第2引数で既知の解が与えられている場合、それ以外の解を探索したい。
    # そのため、「既知の解と完全一致してはいけない」という制約を与える。
    if known_answer is not None:
        if re.fullmatch(r"[1-9]{81}", known_answer) is None:
            raise ValueError("error: known_answer is invalid format.")
        s.add(z3.Not(z3.And([vals[y][x] == int(known_answer[y*9+x]) for y in range(9) for x in range(9)])))

    # 充足可能か調べる。充足可能ならば、答えを得てstrにまとめて返す。
    if s.check() == z3.sat:
        m = s.model()
        sudoku_answer_list = list(sudoku_problem)
        for y in range(9):
            for x in range(9):
                sudoku_answer_list[y*9+x] = m.evaluate(vals[y][x]).as_string()
                if sudoku_problem[y*9+x] != ".":
                    assert sudoku_answer_list[y*9+x] == sudoku_problem[y*9+x]
        return "".join(sudoku_answer_list)

    #充足不可能だったので問題文をそのまま返す。
    return sudoku_problem

def main_func(sudoku_problem: str, solver):

    print_sikiri()
    print(f'solver is "{solver.__name__}"')

    sudoku_answer = solver(sudoku_problem)

    if "." in sudoku_answer:
        print(f"{solver.__name__}: conclusion: the input problem has no solution.")
        return

    another_answer = solver(sudoku_problem, sudoku_answer)

    assert sudoku_answer != another_answer

    if "." in another_answer:
        print(f"{solver.__name__}: conclusion: the input problem has an unique solution.")
        print(f"answer1: {sudoku_answer}")
        print(f"(sha256: {hashlib.sha256(sudoku_answer.encode('utf-8')).hexdigest()})")
        print_sudoku(sudoku_answer)
        return

    print(f"{solver.__name__}: conclusion: the input problem has two or more solutions.")
    print(f"answer1: {sudoku_answer}")
    print(f"(sha256: {hashlib.sha256(sudoku_answer.encode('utf-8')).hexdigest()})")
    print_sudoku(sudoku_answer)
    print(f"answer2: {another_answer}")
    print(f"(sha256: {hashlib.sha256(another_answer.encode('utf-8')).hexdigest()})")
    print_sudoku(another_answer)


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print("error: this program requires just one argument.")
        sys.exit(1)
    sudoku_problem = args[1].strip()
    if is_varid_sudoku_problem_string_format(sudoku_problem) is False:
        print(f"error: the argument string format is invalid.")
        sys.exit(1)

    print("input problem is:")
    print_sudoku(sudoku_problem)

    main_func(sudoku_problem, solve_pulp)
    main_func(sudoku_problem, solve_z3py)
