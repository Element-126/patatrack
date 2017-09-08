#pragma once

#include "CellularAutomaton.hpp"

template<std::size_t maxNumberOfQuadruplets>
class CPUCellularAutomaton
    : public CellularAutomaton
{
public:

    CPUCellularAutomaton() = default; // FIXME

    virtual std::vector<Quadruplet> run(HostEvent event) override;

private:

    // TODO
};
