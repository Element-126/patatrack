#include "CellularAutomaton.hpp"

// HPX_REGISTER_COMPONENT_MODULE();

// using CellularAutomaton_type = hpx::components::simple_component<CellularAutomaton>;
// HPX_REGISTER_COMPONENT(CellularAutomaton_type, CellularAutomaton);

CellularAutomaton::CellularAutomaton(
    Host::Region region,
    Host::Cuts cuts,
    Host::MaxHitsAndLayers max
) :
    region(region),
    thetaCut(cuts.theta),
    phiCut(cuts.phi),
    hardPtCut(cuts.hardPt),
    maxNumberOfHits(max.hits),
    maxNumberOfDoublets(max.doublets),
    maxNumberOfLayers(max.layers),
    maxNumberOfLayerPairs(max.layerPairs),
    maxNumberOfRootLayerPairs(max.rootLayerPairs)
{}
