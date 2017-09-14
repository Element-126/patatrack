#pragma once

#include <hpx/include/components.hpp>

#include <vector>
#include <array>
#include <stdexcept>

#include "../Event/HostEvent.hpp"
#include "../Event/HostRegion.hpp"
#include "HostQuadruplet.hpp"
#include "Cuts.hpp"
#include "MaxHitsAndLayers.hpp"

class CellularAutomaton
    // : public hpx::components::component_base<CellularAutomaton>
{
public:

    CellularAutomaton(
        Host::Region region,
        Host::Cuts cuts,
        Host::MaxHitsAndLayers maxNumbersOfHitsAndLayers
    );

    virtual ~CellularAutomaton() {}

    // Find quadruplets in one event
    virtual std::vector<Host::Quadruplet> run(Host::Event event)
    {
        throw std::runtime_error("Called pure virtual function CellularAutomaton::run()");
    }

protected:

    // Region in phase space
    Host::Region region;

    // Cuts
    float thetaCut;
    float phiCut;
    float hardPtCut;

    // Hits & layer parameters
    unsigned int maxNumberOfHits;
    unsigned int maxNumberOfDoublets;
    unsigned int maxNumberOfLayers;
    unsigned int maxNumberOfLayerPairs;
    unsigned int maxNumberOfRootLayerPairs;
};
