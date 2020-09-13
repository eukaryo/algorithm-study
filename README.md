#### DFPN_single_file.cpp

https://eukaryote.hateblo.jp/entry/2020/05/03/133918

df-pnは詰将棋とかのために考案されたアルゴリズムである。DFPN_single_file.cppではdf-pnでオセロの終盤を解く。性能は普通のオセロソフトより遥かに悪い。実装が悪いというよりもdf-pnはオセロの終盤に向いてないためだと考えている。

#### GMP_Lucas_Lehmer.cpp

https://eukaryote.hateblo.jp/entry/2020/07/11/112648

https://eukaryote.hateblo.jp/entry/2020/08/09/181731

リュカレーマーテスト（メルセンヌ数が素数か判定するアルゴリズム）をGMPで愚直に行うコード。GMPの多倍長乗算は主にショーンハーゲシュトラッセン法だと思われる。このコードは有名な無料ソフトより数倍遅いのだが、それらでは実はショーンハーゲシュトラッセン法を使っておらず、IBDWTというメルセンヌ数に特化した別種のアルゴリズムを使っていた。以降の私の試行の記録はIBDWTフォルダの中にある。

#### algorithm-library.cpp

昔競プロやってたときに手元に持ってたやつ。

#### hash_64bit.cpp

64bit整数を全単射でハッシュする関数が欲しいときのために、Blowfish暗号とSplitMax64疑似乱数生成器を用意しておいた。
SplitMax64の逆変換は元コードになかっので自作した。
