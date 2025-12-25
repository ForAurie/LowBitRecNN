#include "LinearAlgebra.hpp"
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
constexpr int N = 1 << 11;
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
double sigmoidf(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double ff(double a, double x) { return  a / (a + (1 - a) * exp(-x)); }
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

// Net


namespace QNet {
    template<typename T = float, typename Mat = LinearAlgebra::Matrix<T>, auto actFunc = sigmoidf, auto f = ff>
    class Net {
        private:
            std::vector<Mat> weights, biases;
        public:
            size_t getInputSize() const { return weights.front().N(); }
            size_t getOutputSize() const { return weights.back().M(); }
            std::vector<Mat>& getWeights() { return weights; }
            std::vector<Mat>& getBiases() { return biases; }
            const std::vector<Mat>& getWeights() const { return weights; }
            const std::vector<Mat>& getBiases() const { return biases; }
            Net(size_t inputSize, const std::vector<size_t>& Layers): weights(Layers.size()), biases(0) {
                if (haveBiases) biases.resize(Layers.size());
                weights.front().resize(inputSize, Layers.front());
                if (haveBiases) biases.front().resize(1, Layers.front());
                for (size_t i = 1; i < Layers.size(); ++i) {
                    weights[i].resize(Layers[i - 1], Layers[i]);
                    if (haveBiases) biases[i].resize(1, Layers[i]);
                }
            }
            Mat forward(Mat input) const {
                for (size_t i = 0; i < weights.size(); ++i) {
                    Mat next(1, weights[i].M(), actFunc(0));
                    for (size_t j = 0; j < weights[i].N(); j++) {
                        for (size_t k = 0; k < weights[i].M(); k++) {
                            next(0, k) += f(next(0, k), input(0, j) * weights[i](j, k));
                        }
                    }
                    input = next;
                    if (haveBiases) {
                        for (size_t j = 0; j < input.M(); j++) {
                            input(0, j) += f(input(0, j), biases[i](0, j));
                        }
                    }
                }
                return input;
            }
            void open(const std::string& filename) {
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
}

// ----------

using namespace std;
using Mat = LinearAlgebra::Matrix<float>;
#define os cout
void loadData(const string& Path, vector<Mat>& inputs, vector<Mat>& targets) {
    ifstream fin(Path);
    int tmp;
    for (size_t i = 0; i < 10; i++) {
        size_t sz; Mat ans(1, 10, 0), in(1, 28 * 28, 0); ans(0, i) = 1;
        fin >> i >> sz;
        os << "Loading " << sz << " samples for digit " << i << endl;
        while (sz--) {
            for (size_t j = 0; j < 28 * 28; j++)
                fin >> tmp, in(0, j) = tmp;
            inputs.push_back(in);
            targets.push_back(ans);
        }
    }
    fin.close();
}
int predict(const Mat& output) {
    int maxIndex = 0;
    for (size_t i = 1; i < output.M(); i++)
        if (output(0, i) > output(0, maxIndex))
            maxIndex = i;
    return maxIndex;
}

int main() {
    os << "Loading ANN model..." << endl;
    QNet::Net<float, Mat, sigmoidf, ff> nn(784, {256, 64, 10});
    cout << "Path: ";
	string Path;
	cin >> Path;
	cout << endl;
	nn.open(Path);

    os << "Loading Testing data..." << endl;
    vector<Mat> inputs, targets;
    loadData(".\\TestData.txt", inputs, targets);

    os << "Testing..." << endl;
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        correct += (predict(nn.forward(inputs[i])) == predict(targets[i]));
    }
    os << "Testing completed." << endl;
    os << "Accuracy: " << correct << " / " << inputs.size() << " = " << (correct * 100.0 / inputs.size()) << "%" << endl;
    int tmp; while (cin >> tmp);
    return 0;
}

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
