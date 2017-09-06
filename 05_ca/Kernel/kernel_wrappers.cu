#include "kernel_wrappers.hpp"
#include "kernels.hpp"

dim3 arrayToDim3(const std::array<unsigned int, 3> &arr)
{
    return dim3(arr[0], arr[1], arr[2]);
}

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
    )
    {
        const dim3 gs = arrayToDim3(grid_size);
        const dim3 bs = arrayToDim3(block_size);
        kernel_create<<<gs, bs, shared_memory, stream>>>(
            event, gpuDoublets, gpuHitsOnLayers, cells, isOuterHitOfCell,
            foundNtuplets, region, maxNumberOfDoublets, maxNumberOfHits
        );
    }

    void connect(
        std::array<unsigned int, 3> const &grid_size,
        std::array<unsigned int, 3> const &block_size,
        unsigned int shared_memory,
        cudaStream_t stream,
        const GPU::Event *event,
        const GPU::LayerDoublets *gpuDoublets,
        GPU::CACell *cells,
        GPU::SimpleVector<unsigned int> *isOuterHitOfCell,
        const GPU::Region* region,
        const float thetaCut,
        const float phiCut,
        const float hardPtCut,
        unsigned int maxNumberOfDoublets,
        unsigned int maxNumberOfHits
    )
    {
        const dim3 gs = arrayToDim3(grid_size);
        const dim3 bs = arrayToDim3(block_size);
        kernel_connect<<<gs, bs, shared_memory, stream>>>(
            event, gpuDoublets, cells, isOuterHitOfCell, region,
            thetaCut, phiCut, hardPtCut, maxNumberOfDoublets, maxNumberOfHits
        );
    }

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
    )
    {
        const dim3 gs = arrayToDim3(grid_size);
        const dim3 bs = arrayToDim3(block_size);
        kernel_find_ntuplets<<<gs, bs, shared_memory, stream>>>(
            event, gpuDoublets, cells, foundNtuplets, rootLayerPairs,
            minHitsPerNtuplet, maxNumberOfDoublets
        );
    }

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
        )
        {
            const dim3 gs = arrayToDim3(grid_size);
            const dim3 bs = arrayToDim3(block_size);
            kernel_debug<<<gs, bs, shared_memory, stream>>>(
                event, gpuDoublets, gpuHitsOnLayers, cells, isOuterHitOfCell,
                foundNtuplets, region,
                thetaCut, phiCut, hardPtCut,
                maxNumberOfDoublets, maxNumberOfHits
            );
        }
        
        void connect(
            std::array<unsigned int, 3> const &grid_size,
            std::array<unsigned int, 3> const &block_size,
            unsigned int shared_memory,
            cudaStream_t stream,
            const GPU::Event *event,
            const GPU::LayerDoublets *gpuDoublets,
            GPU::CACell *cells,
            GPU::SimpleVector<unsigned int> *isOuterHitOfCell,
            const GPU::Region* region,
            unsigned int maxNumberOfDoublets,
            unsigned int maxNumberOfHits
        )
        {
            const dim3 gs = arrayToDim3(grid_size);
            const dim3 bs = arrayToDim3(block_size);
            kernel_debug_connect<<<gs, bs, shared_memory, stream>>>(
                event, gpuDoublets, cells, isOuterHitOfCell, region,
                maxNumberOfDoublets, maxNumberOfHits
            );
        }

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
        )
        {
            const dim3 gs = arrayToDim3(grid_size);
            const dim3 bs = arrayToDim3(block_size);
            kernel_debug_find_ntuplets<<<gs, bs, shared_memory, stream>>>(
                event, gpuDoublets, cells, foundNtuplets, rootLayerPairs,
                minHitsPerNtuplet, maxNumberOfDoublets
            );
        }

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
        )
        {
            const dim3 gs = arrayToDim3(grid_size);
            const dim3 bs = arrayToDim3(block_size);
            debug_input_data<<<gs, bs, shared_memory, stream>>>(
                event, gpuDoublets, gpuHitsOnLayers,
                region, maxNumberOfHits
            );
        }
    }
}
