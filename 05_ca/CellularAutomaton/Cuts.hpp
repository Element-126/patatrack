#pragma once

#include <hpx/include/serialization.hpp>

namespace Host
{
    struct Cuts
    {
        float theta;
        float phi;
        float hardPt;

        Cuts(
            float thetaCut,
            float phiCut,
            float hardPtCut
        ) :
            theta(thetaCut),
            phi(phiCut),
            hardPt(hardPtCut)
        {}

        template<typename Archive>
        void serialize(Archive& ar, unsigned int version)
        {
            ar & theta & phi & hardPt;
        }
    };
}
