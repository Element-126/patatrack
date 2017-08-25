#pragma once

#include <hpx/include/serialization.hpp>

namespace Host
{
    struct MaxHitsAndLayers
    {
        unsigned int hits;
        unsigned int doublets;
        unsigned int layers;
        unsigned int layerPairs;
        unsigned int rootLayerPairs;

        // [FIXME] Needed by HPX
        MaxHitsAndLayers() = default;

        MaxHitsAndLayers(
            unsigned int maxNumberOfHits,
            unsigned int maxNumberOfDoublets,
            unsigned int maxNumberOfLayers,
            unsigned int maxNumberOfLayerPairs,
            unsigned int maxNumberOfRootLayerPairs
        ) :
            hits(maxNumberOfHits),
            doublets(maxNumberOfDoublets),
            layers(maxNumberOfLayers),
            layerPairs(maxNumberOfLayerPairs),
            rootLayerPairs(maxNumberOfRootLayerPairs)
        {}

        template<typename Archive>
        void serialize(Archive& ar, unsigned int version)
        {
            ar & hits & doublets & layers & layerPairs & rootLayerPairs;
        }
    };
}
