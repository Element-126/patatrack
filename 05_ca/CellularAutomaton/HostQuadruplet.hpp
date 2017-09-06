#pragma once

#include <array>
#include <iostream>

namespace Host
{
    using Quadruplet = std::array<std::array<int, 2>, 3>;
}

inline std::ostream & operator<<(std::ostream & os, Host::Quadruplet const &q)
{
    constexpr auto sep = ", ";
    return os << q[0][0] << sep << q[0][1] << sep
              << q[1][0] << sep << q[1][1] << sep
              << q[2][0] << sep << q[2][1];
}
