#include <bits/stdc++.h>
// 全局常量
using nT = std::uint16_t;
constexpr nT N = 1 << 11;
// ----------

// 类的定义
nT __ADD[N][N], __SUB[N][N], __TIMES[N][N], __DIV[N][N];
template <typename T = std::uint16_t, typename T1 = float, int len = 11, int V = 8>
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
                if (val() == 0 && o.val() == 0) return false;
                return s1 > s2;
            } else if (s1) return val() > o.val();
            else return val() > o.val();
        }
        SignedType& operator=(const SignedType&o) { v = o.v; return *this; }

        SignedType operator+(const SignedType& o) const { return SignedType(__ADD[v][o.v]); }
        SignedType& operator+=(const SignedType& o) { return v = __ADD[v][o.v], *this; }

        SignedType operator-(const SignedType& o) const { return SignedType(__SUB[v][o.v]); }
        SignedType& operator-=(const SignedType& o) { return v = __SUB[v][o.v], *this; }

        SignedType operator*(const SignedType& o) const { return SignedType(__TIMES[v][o.v]); }
        SignedType& operator*=(const SignedType& o) { return v = __TIMES[v][o.v], *this; }

        SignedType operator/(const SignedType& o) const { return SignedType(__DIV[v][o.v]); }
        SignedType& operator/=(const SignedType& o) { return v = __DIV[v][o.v], *this; }
    #undef sign
    #undef MASK
};
// ----------

// 辅助
using BT = SignedType<nT, double, 11, 8>;
nT __SIGMOID[N], __F[N][N];
BT sigmoid(BT x) {
    return BT(__SIGMOID[x.get()]);
}
BT f(BT x, BT y) { return BT(__F[x.get()][y.get()]); }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double f(double a, double x) { return  a / (a + (1 - a) * exp(-x)); }
// ----------

void init() { // 预处理
    for (nT i = 0; i < N; i++) __SIGMOID[i] = BT(sigmoid(BT(i).val())).get();
    for (nT i = 0; i < N; i++) {
        // double v = BT(i).val();
        // if (v < 0 || v > 1) continue;
        for (nT j = 0; j < N; j++)
            __F[i][j] = BT(f(BT(i).val(), BT(j).val())).get();
    }
    for (nT i = 0; i < N; i++)
        for (nT j = 0; j < N; j++)
            __ADD[i][j] = BT(BT(i).val() + BT(j).val()).get();
    for (nT i = 0; i < N; i++)
        for (nT j = 0; j < N; j++)
            __SUB[i][j] = BT(BT(i).val() - BT(j).val()).get();
    for (nT i = 0; i < N; i++)
        for (nT j = 0; j < N; j++)
            __TIMES[i][j] = BT(BT(i).val() * BT(j).val()).get();
    for (nT i = 0; i < N; i++)
        for (nT j = 0; j < N; j++)
            __DIV[i][j] = BT(BT(i).val() / BT(j).val()).get();
}

#include "LinearAlgebra.hpp"
// #include <random>
// #include <iostream>
// #include <utility>
// #include <vector>
// #include <string>
// #include <cstdio>
// #include <cassert>
// #include <cmath>
// #include <ctime>

// 改版 QNet，只有前向传播
template<typename T = float, typename Mat = LinearAlgebra::Matrix<T>, const bool haveBiases = true, const bool haveOutputActivation = true>
class QNet {
    private:
        size_t inputSize, outputSize;
        std::vector<Mat> weights, biases;
        std::vector<std::pair<Mat, Mat>> __forward(Mat input) const {
            std::vector<std::pair<Mat, Mat>> res;
            for (size_t i = 0; i < weights.size() - 1; i++) {
                input *= weights[i];
                if (haveBiases) input += biases[i];
                res.push_back(std::make_pair(input, Mat()));
                input.applyFunctionSelf(actFunc);
                res.back().second = input;
            }
            input *= weights.back();
            if (haveBiases) input += biases.back();
            res.push_back(std::make_pair(input, Mat()));
            if (haveOutputActivation) {
                input.applyFunctionSelf(actFunc);
                res.back().second = input;
            }
            return res;
        }
    public:
        const size_t& getInputSize() const { return inputSize; }
        const size_t& getOutputSize() const { return outputSize; }
        std::vector<Mat>& getWeights() { return weights; }
        std::vector<Mat>& getBiases() { return biases; }
        const std::vector<Mat>& getWeights() const { return weights; }
        const std::vector<Mat>& getBiases() const { return biases; }
        ANN(const size_t& __inputSize, const size_t& __outputSize, const std::vector<size_t>& hiddenLayers): inputSize(__inputSize), outputSize(__outputSize), weights(hiddenLayers.size() + 1), biases(0) {
            if (haveBiases) biases.resize(hiddenLayers.size() + 1);
            weights.front().resize(inputSize, hiddenLayers.front());
            if (haveBiases) biases.front().resize(1, hiddenLayers.front());
            weights.back().resize(hiddenLayers.back(), outputSize);
            if (haveBiases) biases.back().resize(1, outputSize);
            for (size_t i = 1; i < hiddenLayers.size(); ++i) {
                weights[i].resize(hiddenLayers[i - 1], hiddenLayers[i]);
                if (haveBiases) biases[i].resize(1, hiddenLayers[i]);
            }
        }
        Mat forward(Mat input) const {
            for (size_t i = 0; i + 1 < weights.size(); ++i) {
                input *= weights[i];
                if (haveBiases) input += biases[i];
                input.applyFunctionSelf(actFunc);
            }
            input *= weights.back();
            if (haveBiases) input += biases.back();
            if (haveOutputActivation) input.applyFunctionSelf(actFunc);
            return input;
        }
        void open(const std::string& filename = "QANN.model") {
            FILE *fin = fopen(filename.c_str(), "rb");
            char sign;
            unsigned int typeSize;
            fread(&sign, sizeof(char), 1, fin);
            assert(sign == 'Q'); // 检查标志
            fread(&typeSize, sizeof(unsigned int), 1, fin);
            assert(typeSize == sizeof(T)); // 检查数据类型大小
            bool __haveBiases, __haveOutputActivation;
            fread(&__haveBiases, sizeof(bool), 1, fin);
            fread(&__haveOutputActivation, sizeof(bool), 1, fin);
            assert(__haveBiases == haveBiases && __haveOutputActivation == haveOutputActivation); // 检查偏置和输出层激活函数一致性
            fread(&inputSize, sizeof(size_t), 1, fin);
            fread(&outputSize, sizeof(size_t), 1, fin);
            size_t layers;
            fread(&layers, sizeof(size_t), 1, fin);
            weights.resize(layers);
            if (haveBiases) biases.resize(layers);
            T tmp;
            for (auto &x: weights) {
                size_t n, m;
                fread(&n, sizeof(size_t), 1, fin);
                fread(&m, sizeof(size_t), 1, fin);
                x.resize(n, m);
                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < m; j++) {
                        fread(&tmp, sizeof(T), 1, fin);
                        x(i, j) = tmp;
                    }
                }
            }
            if (haveBiases) {
                for (auto &x: biases) {
                    size_t n;
                    fread(&n, sizeof(size_t), 1, fin);
                    x.resize(1, n);
                    for (size_t i = 0; i < n; i++) {
                        fread(&tmp, sizeof(T), 1, fin);
                        x(0, i) = tmp;
                    }
                }
            }
            fclose(fin);
        }
};

int main() {
    init();

    return 0;
}

// TODO:
// template <typename T = std::uint16_t, typename T1 = float, int len = 11, int V = 8>
// class UnsignedType {
//     #define MASK ((1 << len) - 1)
//     private:
//         T v;
//     public:
//         inline T1 val() const { return (T1) V * (T1) v / (T1) MASK; }
//         UnsignedType() : v(0) {}
//         UnsignedType(T v) : v(v) {}
//         UnsignedType(T1 x) : v(std::min(MASK, (T) std::round(x / (T1) V * (T1) MASK))) {}
//         inline T get() const {
//             return v;
//         }
//         inline bool operator==(const UnsignedType& o) const { return v == o.v; }
//         inline bool operator!=(const UnsignedType& o) const { return v != o.v; }
//         inline bool operator<(const UnsignedType& o) const { return v < o.v; }
//     #undef MASK
// };
