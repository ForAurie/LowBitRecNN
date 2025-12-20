#include <bits/stdc++.h>
using nT = std::uint16_t;
constexpr nT N = 1 << 11;
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
using BT = SignedType<nT, double, 11, 8>;
nT __SIGMOID[N], __F[N][N];
BT sigmoid(BT x) {
    return BT(__SIGMOID[x.get()]);
}
BT f(BT x, BT y) { return BT(__F[x.get()][y.get()]); }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double f(double a, double x) { return  a / (a + (1 - a) * exp(-x)); }
int main() {
    using namespace std;
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
