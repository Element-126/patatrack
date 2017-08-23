//   Copyright 2017, Felice Pantaleo, CERN
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
#ifndef RecoPixelVertexing_PixelTriplets_HostHitsAndDoublets_h
#define RecoPixelVertexing_PixelTriplets_HostHitsAndDoublets_h

#include <hpx/include/serialization.hpp>

#include <vector>

struct HostLayerHits
{
    unsigned int layerId;
    size_t size;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    template<typename Archive>
    void serialize(Archive& ar, unsigned int version)
    {
        ar & layerId & size & x & y & z;
    }
};

struct HostLayerDoublets
{
    size_t size;
    unsigned int innerLayerId;
    unsigned int outerLayerId;
    std::vector<unsigned int> indices;

    template<typename Archive>
    void serialize(Archive& ar, unsigned int version)
    {
        ar & size & innerLayerId & outerLayerId & indices;
    }
};

#endif // RecoPixelVertexing_PixelTriplets_HostHitsAndDoublets_h