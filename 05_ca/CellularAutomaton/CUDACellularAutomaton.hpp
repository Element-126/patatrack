#pragma once

#include <hpx/lcos/local/channel.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <tuple>

#include "CellularAutomaton.hpp"
#include "../Event/GPUEvent.h"
#include "../Event/GPURegion.h"
#include "../HitsAndDoublets/GPUHitsAndDoublets.h"
#include "../Vector/GPUSimpleVector.hpp"
#include "GPUQuadruplet.hpp"
#include "GPUCACell.hpp"

class CUDACellularAutomaton :
    public CellularAutomaton,
    public hpx::components::component_base<CUDACellularAutomaton>
{
public:

    // [FIXME] Default constructor, just to make HPX happy
    // Warning: does not construct a valid object, but just a placeholder
    CUDACellularAutomaton() :
        maxNumberOfQuadruplets(3000),
        maxCellsPerHit(100),
        gpuIndex(0),
        nStreams(0),
        eventQueueSize(0)
        {}

    CUDACellularAutomaton(
        unsigned int maxNumberOfQuadruplets,
        unsigned int maxCellsPerHit,
        Host::Region region,
        Host::Cuts cuts,
        Host::MaxHitsAndLayers maxNumbersOfHitsAndLayers,
        unsigned int gpuIndex,
        unsigned int nStreams,
        unsigned int eventQueueSize
    );

    virtual ~CUDACellularAutomaton() override;

    // Run the CUDA implementation of the CA to process one event
    virtual std::vector<Host::Quadruplet> run(Host::Event event) override;

    // HPX actions
    HPX_DEFINE_COMPONENT_ACTION(CUDACellularAutomaton, run, run_action);

private:

    // Constants
    const unsigned int maxNumberOfQuadruplets;
    const unsigned int maxCellsPerHit;

    // Device
    const unsigned int gpuIndex;

    // CUDA streams
    const unsigned int nStreams;
    std::vector<cudaStream_t> streams;
    hpx::lcos::local::channel<unsigned int> streamQueue;
    // Data member passed to callbacks, used to identify calling stream
    std::vector<std::tuple<CUDACellularAutomaton*, unsigned int, unsigned int>> streamInfo;

    // Work queue, preallocated on the host
    const unsigned int eventQueueSize;
    hpx::lcos::local::channel<unsigned int> resourceQueue;

    // Vector of one-element channels (one per CUDA stream) set by the CUDA
    // callback to notify the corresponding suspended HPX thread when
    // asynchronous operations have completed
    std::vector<hpx::lcos::local::one_element_channel<void>> kernelsDone;

    // Host-side pinned memory for the input data
    GPU::Region *h_regionParams;
    GPU::Event *h_events;
    unsigned int *h_indices;
    GPU::LayerDoublets *h_doublets;
    GPU::LayerHits *h_layers;
    float *h_x, *h_y, *h_z;
    unsigned int *h_rootLayerPairs;

    // Host-side pinned memory for the output data
    std::vector<GPU::SimpleVector<GPU::Quadruplet>*> h_foundNtuplets;

    // Temporary buffers, used to copy objects containing pointers between host and device
    std::vector<GPU::LayerHits*> tmp_layers;
    std::vector<GPU::LayerDoublets*> tmp_layerDoublets;
    std::vector<GPU::SimpleVector<GPU::Quadruplet>*> tmp_foundNtuplets; // NEW
    std::vector<GPU::SimpleVector<unsigned int>*> tmp_isOuterHitOfCell; // NEW

    // Device-side memory buffers, used by the kernels
    GPU::Region *d_regionParams;
    GPU::Event *d_events;
    unsigned int *d_indices;
    GPU::LayerDoublets *d_doublets;
    GPU::LayerHits *d_layers;
    float *d_x, *d_y, *d_z;
    unsigned int *d_rootLayerPairs;

    // Device-side allocations for intermediate results
    GPU::CACell *device_theCells;
    std::vector<GPU::SimpleVector<unsigned int>*> device_isOuterHitOfCell;

    // Device-side memory for the results
    std::vector<GPU::SimpleVector<GPU::Quadruplet>*> d_foundNtuplets;

    // Resource management
    void cleanup();
    void initialize_globals();
    void asyncResetCAState(unsigned int streamIndex);

    bool allocateHostMemory();
    void freeHostMemory();

    bool createCUDAStreams();
    void destroyCUDAStreams();

    // Must be called *after* `allocateHostMemory()` (requires valid temporaries `tmp_*`)
    bool allocateDeviceMemory();
    // Must be called *before* `freeHostMemory()` (requires valid temporaries `tmp_*`)
    void freeDeviceMemory();

    void createChannels();
    void destroyChannels();

    // Copy events from/to pinned buffers and GPU
    void copyEventToPinnedBuffers(const Host::Event& evt, unsigned int idx);
    void asyncCopyEventToGPU(unsigned int bufferIndex, unsigned int streamIndex);
    void asyncCopyResultsToHost(unsigned int streamIndex, unsigned int bufferIndex);

    // Setup CUDA callback to wake-up the calling thread when asynchronous
    // operations have completed
    void enqueueCallback(unsigned int streamIndex);

    // Copy GPU quadruplets to a vector of host quadruplets
    std::vector<Host::Quadruplet> makeQuadrupletVector(unsigned int bufferIndex) const;
};

using CUDACellularAutomaton_run_action = CUDACellularAutomaton::run_action;
HPX_REGISTER_ACTION_DECLARATION(CUDACellularAutomaton_run_action);
