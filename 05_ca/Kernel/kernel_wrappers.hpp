//
// Contains wrappers to expose a C++ interface for CUDA kernels
//

#if !defined(KERNEL_WRAPPERS_HPP)
#define KERNEL_WRAPPERS_HPP

#include <cuda.h>

#include <array>

#include "../Event/GPUEvent.h"
#include "../Event/GPURegion.h"
#include "../HitsAndDoublets/GPUHitsAndDoublets.h"
#include "../CellularAutomaton/GPUCACell.hpp"
#include "../CellularAutomaton/GPUQuadruplet.hpp"
#include "../Vector/GPUSimpleVector.hpp"

namespace kernel
{
    void create(
        std::array<unsigned int, 3> const &grid_size,
        std::array<unsigned int, 3> const &block_size,
        unsigned int shared_memory,
        cudaStream_t stream,
        const GPU::Event *event,
        const GPU::LayerDoublets *gpuDoublets,
        const GPU::LayerHits *gpuHitsOnLayers,
        GPU::CACell *cells,
        GPU::SimpleVector<unsigned int> *isOuterHitOfCell,
        GPU::SimpleVector<GPU::Quadruplet> *foundNtuplets,
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
        GPU::SimpleVector<unsigned int> *isOuterHitOfCell,
        const GPU::Region *region,
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
        GPU::SimpleVector<GPU::Quadruplet> *foundNtuplets,
        unsigned int *rootLayerPairs,
        unsigned int minHitsPerNtuplet,
        unsigned int maxNumberOfDoublets
    );

    namespace debug
    {
        void all(
            std::array<unsigned int, 3> const &grid_size,
            std::array<unsigned int, 3> const &block_size,
            unsigned int shared_memory,
            cudaStream_t stream,
            const GPU::Event *event,
            const GPU::LayerDoublets *gpuDoublets,
            const GPU::LayerHits *gpuHitsOnLayers,
            GPU::CACell *cells,
            GPU::SimpleVector<unsigned int> *isOuterHitOfCell,
            GPU::SimpleVector<GPU::Quadruplet> *foundNtuplets,
            const GPU::Region *region,
            const float thetaCut,
            const float phiCut,
            const float hardPtCut,
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
            GPU::SimpleVector<unsigned int> *isOuterHitOfCell,
            const GPU::Region *region,
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
            GPU::SimpleVector<GPU::Quadruplet> *foundNtuplets,
            unsigned int *rootLayerPairs,
            unsigned int minHitsPerNtuplet,
            unsigned int maxNumberOfDoublets
        );

        void input_data(
            std::array<unsigned int, 3> const &grid_size,
            std::array<unsigned int, 3> const &block_size,
            unsigned int shared_memory,
            cudaStream_t stream,
            const GPU::Event *event,
            const GPU::LayerDoublets *gpuDoublets,
            const GPU::LayerHits *gpuHitsOnLayers,
            const GPU::Region *region,
            unsigned int maxNumberOfHits
        );
    }
}

#endif // !defined(KERNEL_WRAPPERS_HPP)
