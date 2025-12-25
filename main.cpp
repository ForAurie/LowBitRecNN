#include <random>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <string>
#include <fstream>

// 全局常量
using nT = std::uint16_t;
constexpr int LEN = 13, N = 1 << LEN;
// ----------

// 类的定义
template <typename T = std::uint16_t, typename T1 = float, int len = 12, int V = 5>
class SignedType {
    #define sign (v >> (len - 1))
    #define MASK ((T(1) << (len - 1)) - 1)
    private:
        T v;
        constexpr T __val() const { return v & MASK; }
    public:
        constexpr T1 val() const {
            if (sign) return -((T1) V * (T1) __val() / (T1) MASK);
            else return (T1) V * (T1) __val() / (T1) MASK;
        }
        constexpr SignedType(): v(0) {}
        constexpr SignedType(T v) : v(v) {}
        constexpr SignedType(const SignedType& o) = default;
        constexpr SignedType(T1 x) : v(x >= T1(0) ? T(0) : T(1) << (len - 1)) {
            if (x < T1(0)) x = -x;
            v |= std::min((T) MASK, (T) std::round(x / (T1) V * (T1) MASK));
        }
        constexpr T get() const { return v; }
        inline bool operator==(const SignedType& o) const { return v == o.v; }
        inline bool operator!=(const SignedType& o) const { return v != o.v; }
        inline bool operator<(const SignedType& o) const {
            bool s1 = v >> (len - 1), s2 = o.v >> (len - 1);
            if (s1 != s2) {
                if (__val() == 0 && o.__val() == 0) return false;
                return s1 > s2;
            } else if (s1) return __val() > o.__val();
            else return __val() < o.__val();
        }
        SignedType& operator=(const SignedType&o) { v = o.v; return *this; }

        // SignedType operator+(const SignedType& o) const { return SignedType(__ADD[v][o.v]); }
        // SignedType& operator+=(const SignedType& o) { return v = __ADD[v][o.v], *this; }

        // SignedType operator-(const SignedType& o) const { return SignedType(__SUB[v][o.v]); }
        // SignedType& operator-=(const SignedType& o) { return v = __SUB[v][o.v], *this; }

        // SignedType operator*(const SignedType& o) const { return SignedType(__TIMES[v][o.v]); }
        // SignedType& operator*=(const SignedType& o) { return v = __TIMES[v][o.v], *this; }

        // SignedType operator/(const SignedType& o) const { return SignedType(__DIV[v][o.v]); }
        // SignedType& operator/=(const SignedType& o) { return v = __DIV[v][o.v], *this; }
    #undef sign
    #undef MASK
};
// ----------

// 函数和数据的定义
using T01 = SignedType<nT, double, LEN, 1>;
using T05 = SignedType<nT, double, LEN, 5>;

nT __SIGMOID01[N], __F[N][N], __TIMES[N][N];

T01 sigmoid(T01 x) { return T01(__SIGMOID01[x.get()]); }
T01 f(T01 x, T05 y) { return T01(__F[x.get()][y.get()]); }
double sigmoidf(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double ff(double a, double x) { return  a / (a + (1.0 - a) * exp(-x)); }
T05 times0105(T01 x, T05 y) { return T05(__TIMES[x.get()][y.get()]); }
// ----------

#include "LinearAlgebra.hpp"

void init() { // 预处理
    for (nT i = 0; i < N; i++) __SIGMOID01[i] = T01(sigmoidf(T01(i).val())).get();
    for (nT i = 0; i < N; i++) {
        for (nT j = 0; j < N; j++)
            __F[i][j] = T01(ff(T01(i).val(), T05(j).val())).get();
    }
    for (nT i = 0; i < N; i++)
        for (nT j = 0; j < N; j++)
            __TIMES[i][j] = T05(T01(i).val() * T05(j).val()).get();
}

// Net
using Mat01 = LinearAlgebra::Matrix<T01>;
using Mat05 = LinearAlgebra::Matrix<T05>;

namespace QNet {
    std::vector<Mat05> weights, biases;
    Mat01 forward(Mat01 input) {
        // BT mx(-7.0), mn(7.0);
        for (size_t i = 0; i < weights.size(); ++i) {
            Mat01 next(1, weights[i].M(), sigmoid(T01(0.0)));
            for (size_t j = 0; j < weights[i].N(); j++) {
                for (size_t k = 0; k < weights[i].M(); k++) {
                    next(0, k) = f(next(0, k), times0105(input(0, j), weights[i](j, k)));
                    // mn = std::min(mn, weights[i](j, k));
                    // mx = std::max(mx, weights[i](j, k));
                }
            }
            input = next;
            for (size_t j = 0; j < input.M(); j++) {
                input(0, j) = f(input(0, j), biases[i](0, j));
                // mn = std::min(mn, biases[i](0, j));
                // mx = std::max(mx, biases[i](0, j));
            }
        }
        // std::cout << "Activation range: [" << mn.val() << ", " << mx.val() << "]" << std::endl;
        // exit(0);
        return input;
    }
    void open(const std::string& filename) {
        FILE *fin = fopen(filename.c_str(), "rb");
        char sign;
        unsigned int typeSize;
        fread(&sign, sizeof(char), 1, fin);
        assert(sign == 'Q'); // 检查标志
        fread(&typeSize, sizeof(unsigned int), 1, fin);
        // assert(typeSize == sizeof(T)); // 检查数据类型大小
        bool __haveBiases, __haveOutputActivation;
        fread(&__haveBiases, sizeof(bool), 1, fin);
        fread(&__haveOutputActivation, sizeof(bool), 1, fin);
        // assert(__haveBiases == haveBiases && __haveOutputActivation == haveOutputActivation); // 检查偏置和输出层激活函数一致性
        size_t layers;
        fread(&layers, sizeof(size_t), 1, fin);
        weights.resize(layers);
        // if (haveBiases)
        biases.resize(layers);
        float tmp;
        for (auto &x: weights) {
            size_t n, m;
            fread(&n, sizeof(size_t), 1, fin);
            fread(&m, sizeof(size_t), 1, fin);
            x.resize(n, m);
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < m; j++) {
                    fread(&tmp, sizeof(float), 1, fin);
                    // x(i, j) = tmp;
                    x(i, j) = T05(tmp);
                }
            }
        }
        // if (haveBiases) {
            for (auto &x: biases) {
                size_t n;
                fread(&n, sizeof(size_t), 1, fin);
                x.resize(1, n);
                for (size_t i = 0; i < n; i++) {
                    fread(&tmp, sizeof(float), 1, fin);
                    // x(0, i) = tmp;
                    x(0, i) = T05(tmp);
                }
            }
        // }
        fclose(fin);
    }
}
// ----------

using namespace std;
#define os cout

// 数据加载和预测
void loadData(const string& Path, vector<Mat01>& inputs, vector<Mat01>& targets) {
    ifstream fin(Path);
    int tmp;
    for (size_t i = 0; i < 10; i++) {
        size_t sz; Mat01 ans(1, 10), in(1, 28 * 28); ans(0, i) = T01((float) 1);
        fin >> i >> sz;
        os << "Loading " << sz << " samples for digit " << i << endl;
        while (sz--) {
            for (size_t j = 0; j < 28 * 28; j++)
                fin >> tmp, in(0, j) = T01((double) tmp);
            inputs.push_back(in);
            targets.push_back(ans);
        }
    }
    fin.close();
}
int predict(const Mat01& output) {
    int maxIndex = 0;
    // for (size_t i = 0; i < output.M(); i++)
    //     os << round(output(0, i).val()) << " ";
    // os << endl;
    for (size_t i = 1; i < output.M(); i++)
        if (output(0, maxIndex) < output(0, i))
            maxIndex = i;
    return maxIndex;
}
// ----------

int main() {
    init();
    os << "Loading ANN model..." << endl;
	string Path;
    Path = "..\\example2.model";
	QNet::open(Path);

    os << "Loading Testing data..." << endl;
    vector<Mat01> inputs, targets;
    loadData("..\\TestData.txt", inputs, targets);

    os << "Testing..." << endl;
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        correct += (predict(QNet::forward(inputs[i])) == predict(targets[i]));
    }
    os << "Testing completed." << endl;
    os << "Accuracy: " << correct << " / " << inputs.size() << " = " << (correct * 100.0 / inputs.size()) << "%" << endl;
    return 0;
}