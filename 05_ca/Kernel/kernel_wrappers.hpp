//
// Contains wrappers to expose a C++ interface for CUDA kernels
//

#if !defined(KERNEL_WRAPPERS_HPP)
#define KERNEL_WRAPPERS_HPP

#include <cuda.h>

#include <array>

#include "Event/GPUEvent.h"
#include "Event/GPURegion.h"
#include "HitsAndDoublets/GPUHitsAndDoublets.h"
#include "CellularAutomaton/GPUCACell.hpp"
#include "CellularAutomaton/GPUQuadruplet.hpp"
#include "GPUSimpleVector.hpp"

namespace kernel
{
    constexpr unsigned int maxNumberOfQuadruplets = 3000;

    void create(
        std::array<unsigned int, 3> const &grid_size,
        std::array<unsigned int, 3> const &block_size,
        unsigned int shared_memory,
        cudaStream_t stream,
        const GPU::Event *event,
        const GPU::LayerDoublets *gpuDoublets,
        const GPU::LayerHits *gpuHitsOnLayers,
        GPU::CACell *cells,
        GPU::SimpleVector<100, unsigned int> *isOuterHitOfCell,
        GPU::SimpleVector<maxNumberOfQuadruplets, GPU::Quadruplet> *foundNtuplets,
        const GPU::Region *region,
        unsigned int maxNumberOfDoublets,
        unsigned int maxNumberOfHits
    );

    void connect(
        std::array<unsigned int, 3> const &grid_size,
        std::array<unsigned int, 3> const &block_size,
        unsigned int shared_memory,
        cudaStream_t stream,
        const GPU::Event *event,
        const GPU::LayerDoublets *gpuDoublets,
        GPU::CACell *cells,
        GPU::SimpleVector<100, unsigned int> *isOuterHitOfCell,
        const GPU::Region* region,
        const float thetaCut,
        const float phiCut,
        const float hardPtCut,
        unsigned int maxNumberOfDoublets,
        unsigned int maxNumberOfHits
    );

    void find_ntuplets(
        std::array<unsigned int, 3> const &grid_size,
        std::array<unsigned int, 3> const &block_size,
        unsigned int shared_memory,
        cudaStream_t stream,
        const GPU::Event *event,
        const GPU::LayerDoublets *gpuDoublets,
        GPU::CACell *cells,
        GPU::SimpleVector<maxNumberOfQuadruplets, GPU::Quadruplet> *foundNtuplets,
        unsigned int *rootLayerPairs,
        unsigned int minHitsPerNtuplet,
        unsigned int maxNumberOfDoublets
    );
}

#endif // !defined(KERNEL_WRAPPERS_HPP)
