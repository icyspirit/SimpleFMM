#ifndef EXPQUAD_HPP
#define EXPQUAD_HPP


#include <array>


// Cheng, Greengard and Rokhlin, J. Comput. Phys. 155, 468 (1999), Tables XIV-XVI.
// Nodes and weights of the outer integral in the plane wave representation of 1/r,
// and the number of trapezoidal points M_k of the inner integral for each node.

template<int digits>
struct ExpQuad;


template<>
struct ExpQuad<3> {
    static constexpr int s = 8;
    static constexpr std::array<double, s> lambda{
        0.10934746769000, 0.51769741015341, 1.13306591611192, 1.88135015110740,
        2.71785409601205, 3.61650274907449, 4.56271053303821, 5.54900885348528};
    static constexpr std::array<double, s> w{
        0.27107502662774, 0.52769158843946, 0.69151504413879, 0.79834400406452,
        0.87164160121354, 0.92643839116924, 0.97294622259483, 1.02413865844686};
    static constexpr std::array<int, s> M{4, 8, 16, 16, 24, 24, 8, 4};
};


template<>
struct ExpQuad<6> {
    static constexpr int s = 17;
    static constexpr std::array<double, s> lambda{
         0.05599002531749,  0.28485138101968,  0.66535365065853,  1.16667904805296,
         1.76443027413431,  2.44029832236380,  3.18032180991515,  3.97371715777193,
         4.81216799410634,  5.68932314511487,  6.60040479444377,  7.54190497469911,
         8.51136569298099,  9.50723242759128, 10.52874809650967, 11.57587019602884,
        12.65078163968520};
    static constexpr std::array<double, s> w{
        0.14239483712194, 0.31017671029271, 0.44557516683709, 0.55303383994159,
        0.63944903363523, 0.70997911214019, 0.76828253949732, 0.81713201141707,
        0.85872191623337, 0.89480789582390, 0.92680189417317, 0.95586282708096,
        0.98299145008230, 1.00913395385703, 1.03531774600508, 1.06318427913963,
        1.10232109521088};
    static constexpr std::array<int, s> M{
        8, 8, 16, 16, 24, 32, 32, 32, 48, 48, 48, 48, 48, 48, 48, 8, 4};
};


template<>
struct ExpQuad<9> {
    static constexpr int s = 26;
    static constexpr std::array<double, s> lambda{
         0.03705701953816,  0.19219683859955,  0.46045971214897,  0.82805130101422,
         1.28121229944787,  1.80792019276297,  2.39814728074333,  3.04359012306582,
         3.73732742924096,  4.47354768940212,  5.24735518169467,  6.05462948620944,
         6.89191648795972,  7.75633860708838,  8.64551915195994,  9.55751929613924,
        10.49078760616705, 11.44412262341269, 12.41664955395045, 13.40781311788324,
        14.41739038894472, 15.44553016867884, 16.49282861241170, 17.56045648926099,
        18.65046484106274, 19.76847686619416};
    static constexpr std::array<double, s> w{
        0.09473396337900, 0.21384206006426, 0.32031528543989, 0.41254929390710,
        0.49176691815621, 0.55998309037174, 0.61909314036708, 0.67064351982741,
        0.71586567032066, 0.75576118553096, 0.79116885492295, 0.82280556212477,
        0.85129012269433, 0.87715909928110, 0.90087981520398, 0.92286282936149,
        0.94347471535979, 0.96305166489156, 0.98191478773972, 1.00038891281291,
        1.01882849188686, 1.03765781507554, 1.05744113465683, 1.07903824697122,
        1.10434337868208, 1.14488166506896};
    static constexpr std::array<int, s> M{
        8, 16, 16, 16, 24, 32, 32, 32, 48, 48, 48, 64, 64, 64, 64, 72,
        72, 80, 88, 88, 88, 88, 88, 72, 32, 4};
};


// Each table has an error floor that no expansion order can get below, so the
// table has to be chosen against the truncation error at p.  Measured relative
// L2 error of the far field against direct summation:
//
//        p       4        8       10       12       13       14       16
//   RTR      8.9e-4   1.9e-5   2.2e-6   6.2e-7   3.2e-7   1.1e-7   2.9e-8
//   3-digit  1.3e-3   8.4e-4   8.4e-4   8.4e-4        -        -   8.4e-4
//   6-digit  8.9e-4   1.9e-5   2.2e-6   7.3e-7   4.3e-7   2.9e-7   4.6e-7
//   9-digit  8.9e-4   1.9e-5   2.2e-6   6.2e-7   3.2e-7        -   3.5e-8
//
// The boundaries below keep the loss under 1.2x.  9-digit shows no floor
// through p = 24.
constexpr int exp_digits_of(int p) noexcept
{
    return p <= 3 ? 3 : (p <= 12 ? 6 : 9);
}


// Only half of each inner ring is stored: the coefficients of the opposite
// point are the complex conjugates, because M(n, -m) = conj(M(n, m)).
template<int digits>
constexpr int exp_size() noexcept
{
    int n = 0;
    for (int k=0; k<ExpQuad<digits>::s; ++k) {
        n += ExpQuad<digits>::M[k]/2;
    }

    return n;
}


#endif
