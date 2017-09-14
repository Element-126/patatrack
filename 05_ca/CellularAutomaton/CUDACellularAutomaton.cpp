#include <stdexcept>
#include <cassert>
// DEBUG
#include <hpx/include/iostreams.hpp>

#include "CUDACellularAutomaton.hpp"

#include "../Kernel/kernel_wrappers.hpp"

HPX_REGISTER_COMPONENT_MODULE();

using CUDACellularAutomaton_type = hpx::components::simple_component<CUDACellularAutomaton>;
HPX_REGISTER_COMPONENT(CUDACellularAutomaton_type, CUDACellularAutomaton);

HPX_REGISTER_ACTION(CUDACellularAutomaton_run_action);

constexpr auto ok = "\033[1;32mOK\033[0m";
constexpr auto warning = "\033[1;33mwarning:\033[0m ";
constexpr auto failed = "\033[1;31mFAILED!\033[0m";

void debugSynchronizeAndCheck(cudaStream_t stream)
{
#ifdef DEBUG
    cudaStreamSynchronize(stream);
    auto const err = cudaGetLastError();
    if (err != cudaSuccess) {
        HPX_THROW_EXCEPTION(hpx::no_success, (cudaGetErrorName(err)), (cudaGetErrorString(err)));
    }
#endif
}

CUDACellularAutomaton::CUDACellularAutomaton(
    unsigned int maxNumberOfQuadruplets,
    unsigned int maxCellsPerHit,
    Host::Region region,
    Host::Cuts cuts,
    Host::MaxHitsAndLayers maxNumbersOfHitsAndLayers,
    unsigned int gpuIndex,
    unsigned int nStreams,
    unsigned int eventQueueSize
) :
    CellularAutomaton(region, cuts, maxNumbersOfHitsAndLayers),
    maxNumberOfQuadruplets(maxNumberOfQuadruplets),
    maxCellsPerHit(maxCellsPerHit),
    gpuIndex(gpuIndex),
    nStreams(nStreams),
    streams(),
    streamQueue(),
    streamInfo(),
    eventQueueSize(eventQueueSize),
    resourceQueue(),
    kernelsDone(nStreams),
    h_regionParams(nullptr),
    h_events(nullptr),
    h_indices(nullptr),
    h_doublets(nullptr),
    h_layers(nullptr),
    h_x(nullptr),
    h_y(nullptr),
    h_z(nullptr),
    h_rootLayerPairs(nullptr),
    h_foundNtuplets(eventQueueSize, nullptr),
    tmp_layers(nStreams, nullptr),
    tmp_layerDoublets(nStreams, nullptr),
    tmp_foundNtuplets(nStreams, nullptr),
    tmp_isOuterHitOfCell(nStreams, nullptr),
    d_regionParams(nullptr),
    d_events(nullptr),
    d_indices(nullptr),
    d_doublets(nullptr),
    d_layers(nullptr),
    d_x(nullptr),
    d_y(nullptr),
    d_z(nullptr),
    d_rootLayerPairs(nullptr),
    device_theCells(nullptr),
    device_isOuterHitOfCell(nStreams, nullptr),
    d_foundNtuplets(nStreams, nullptr)
{
    // Set GPU
    // hpx::cerr << "Setting device " << gpuIndex << "... " << hpx::flush;
    cudaSetDevice(gpuIndex);
    // hpx::cerr << ok << hpx::endl << hpx::flush;

    // Allocate resources, or throw exception on failure
    if (!allocateHostMemory()) {
        throw std::runtime_error("Could not allocate CUDA pinned memory.");
    }

    if (!allocateDeviceMemory()) {
        throw std::runtime_error("Could not allocate device memory.");
    }

    if (!createCUDAStreams()) {
        throw std::runtime_error("Failed to create CUDA streams.");
    }

    // Copy global constants to GPU
    initialize_globals();

    // Initialize HPX channels
    createChannels();
}


CUDACellularAutomaton::~CUDACellularAutomaton()
{
    cudaDeviceSynchronize();

    //std::cerr << "Cleaning up resources... ";
    cleanup();
    //std::cerr << ok << std::endl;

    //std::cerr << warning << "Destroying CA NOW!" << std::endl;
}

void CUDACellularAutomaton::cleanup()
{
    // Free resources in the reverse order of the allocation
    // This is due to host objects holding device pointers
    destroyCUDAStreams();
    destroyChannels();
    freeDeviceMemory();
    freeHostMemory();
}

void CUDACellularAutomaton::initialize_globals()
{
    cudaMemcpy(
        d_regionParams, h_regionParams, sizeof(GPU::Region), cudaMemcpyHostToDevice
    );
}

// Allocate (pinned) memory buffers on the host
bool CUDACellularAutomaton::allocateHostMemory()
{
    // Host-side buffers
    bool success = cudaMallocHost(
        &h_regionParams,
        sizeof(GPU::Region)
    ) == cudaSuccess
    && cudaMallocHost(
        &h_events,
        eventQueueSize * sizeof(GPU::Event)
    ) == cudaSuccess
    && cudaMallocHost(
        &h_indices,
        eventQueueSize * maxNumberOfLayerPairs * maxNumberOfDoublets * 2 * sizeof(unsigned int)
    ) == cudaSuccess
    && cudaMallocHost(
        &h_doublets,
        eventQueueSize * maxNumberOfLayerPairs * sizeof(GPU::LayerDoublets)
    ) == cudaSuccess
    && cudaMallocHost(
        &h_layers,
        eventQueueSize * maxNumberOfLayers * sizeof(GPU::LayerHits)
    ) == cudaSuccess
    && cudaMallocHost(
        &h_x,
        eventQueueSize * maxNumberOfLayers * maxNumberOfHits * sizeof(float)
    ) == cudaSuccess
    && cudaMallocHost(
        &h_y,
        eventQueueSize * maxNumberOfLayers * maxNumberOfHits * sizeof(float)
    ) == cudaSuccess
    && cudaMallocHost(
        &h_z,
        eventQueueSize * maxNumberOfLayers * maxNumberOfHits * sizeof(float)
    ) == cudaSuccess
    && cudaMallocHost(
        &h_rootLayerPairs,
        eventQueueSize * maxNumberOfRootLayerPairs * sizeof(unsigned int)
    ) == cudaSuccess;

    // Instantiate host objects on pinned memory
    new (h_regionParams) GPU::Region{
        region.ptmin,
        region.region_origin_x,
        region.region_origin_y,
        region.region_origin_radius
    };

    using QuadrupletVector = GPU::SimpleVector<GPU::Quadruplet>;
    for (unsigned int i = 0 ; i < eventQueueSize ; ++i) {
        QuadrupletVector *obj_ptr = nullptr;
        GPU::Quadruplet *data_ptr = nullptr;
        // Allocation for struct...
        success = success && cudaMallocHost(
            &obj_ptr,
            sizeof(QuadrupletVector)
        ) == cudaSuccess
        // ... and for its data.
        && cudaMallocHost(
            &data_ptr,
            maxNumberOfQuadruplets * sizeof(GPU::Quadruplet)
        ) == cudaSuccess;
        // In-place construction of the object itself
        h_foundNtuplets[i] =
            new (obj_ptr) QuadrupletVector(maxNumberOfQuadruplets, data_ptr);
    }

    // Temporaries
    assert(tmp_layers.size() == nStreams);
    assert(tmp_layerDoublets.size() == nStreams);
    assert(tmp_foundNtuplets.size() == nStreams);
    assert(tmp_isOuterHitOfCell.size() == nStreams);
    for (unsigned int i = 0 ; i < nStreams ; ++i) {
        success = success && cudaMallocHost(
            &tmp_layers[i],
            maxNumberOfLayers * sizeof(GPU::LayerHits)
        ) == cudaSuccess
        && cudaMallocHost(
            &tmp_layerDoublets[i],
            maxNumberOfLayerPairs * sizeof(GPU::LayerDoublets)
        ) == cudaSuccess
        // Allocate memory only for the object, not the data
        && cudaMallocHost(
            &tmp_foundNtuplets[i],
            sizeof(QuadrupletVector)
        ) == cudaSuccess
        // This one is a dense 2D array of vectors !
        // The vectors are themselves contiguous in memory
        && cudaMallocHost(
            &tmp_isOuterHitOfCell[i],
            maxNumberOfLayers * maxNumberOfHits * sizeof(GPU::SimpleVector<unsigned int>)
        ) == cudaSuccess;
    }

    if (!success)
    {
        // At least one allocation failed -> clean-up resources
        cleanup();
    } else {
        // Allocation succeeded -> assert that the pointers are valid
        assert(h_regionParams);
        assert(h_events);
        assert(h_indices);
        assert(h_doublets);
        assert(h_layers);
        assert(h_x);
        assert(h_y);
        assert(h_z);
        assert(h_rootLayerPairs);
        for (unsigned int i = 0 ; i < eventQueueSize ; ++i) {
            assert(h_foundNtuplets[i]);
            assert(h_foundNtuplets[i]->m_data);
        }
        for (unsigned int i = 0 ; i < nStreams ; ++i) {
            assert(tmp_layers[i]);
            assert(tmp_layerDoublets[i]);
            assert(tmp_foundNtuplets[i]);
            assert(tmp_isOuterHitOfCell[i]);
        }
    }
    return success;
}

void CUDACellularAutomaton::freeHostMemory()
{
    cudaFreeHost(h_regionParams); h_regionParams = nullptr;
    cudaFreeHost(h_events); h_events = nullptr;
    cudaFreeHost(h_indices); h_indices = nullptr;
    cudaFreeHost(h_doublets); h_doublets = nullptr;
    cudaFreeHost(h_layers); h_layers = nullptr;
    cudaFreeHost(h_x); h_x = nullptr;
    cudaFreeHost(h_y); h_y = nullptr;
    cudaFreeHost(h_z); h_z = nullptr;
    cudaFreeHost(h_rootLayerPairs); h_rootLayerPairs = nullptr;
    assert(h_foundNtuplets.size() == eventQueueSize);
    for (unsigned int i = 0 ; i < eventQueueSize ; ++i) {
        cudaFreeHost(h_foundNtuplets[i]->m_data); h_foundNtuplets[i]->m_data = nullptr;
        cudaFreeHost(h_foundNtuplets[i]); h_foundNtuplets[i] = nullptr;
    }
    assert(tmp_layers.size() == nStreams);
    assert(tmp_layerDoublets.size() == nStreams);
    assert(tmp_foundNtuplets.size() == nStreams);
    assert(tmp_isOuterHitOfCell.size() == nStreams);
    for (unsigned int i = 0 ; i < nStreams ; ++i) {
        cudaFreeHost(tmp_layers[i]); tmp_layers[i] = nullptr;
        cudaFreeHost(tmp_layerDoublets[i]); tmp_layerDoublets[i] = nullptr;
        cudaFreeHost(tmp_foundNtuplets[i]); tmp_foundNtuplets[i] = nullptr;
        cudaFreeHost(tmp_isOuterHitOfCell[i]); tmp_isOuterHitOfCell[i] = nullptr;
    }
}


bool CUDACellularAutomaton::allocateDeviceMemory()
{
    bool success = cudaMalloc(
        &d_regionParams,
        sizeof(GPU::Region)
    ) == cudaSuccess
    && cudaMalloc(
        &d_events,
        nStreams * sizeof(GPU::Event)
    ) == cudaSuccess
    && cudaMalloc(
        &d_indices,
        nStreams * maxNumberOfLayerPairs * maxNumberOfDoublets * 2 * sizeof(unsigned int)
    ) == cudaSuccess
    && cudaMalloc(
        &d_doublets,
        nStreams * maxNumberOfLayerPairs * sizeof(GPU::LayerDoublets)
    ) == cudaSuccess
    && cudaMalloc(
        &d_layers,
        nStreams * maxNumberOfLayers * sizeof(GPU::LayerHits)
    ) == cudaSuccess
    && cudaMalloc(
        &d_x,
        nStreams * maxNumberOfLayers * maxNumberOfHits * sizeof(float)
    ) == cudaSuccess
    && cudaMalloc(
        &d_y,
        nStreams * maxNumberOfLayers * maxNumberOfHits * sizeof(float)
    ) == cudaSuccess
    && cudaMalloc(
        &d_z,
        nStreams * maxNumberOfLayers * maxNumberOfHits * sizeof(float)
    ) == cudaSuccess
    && cudaMalloc(
        &d_rootLayerPairs,
        nStreams * maxNumberOfRootLayerPairs * sizeof(unsigned int)
    ) == cudaSuccess
    && cudaMalloc(
        &device_theCells,
        nStreams * maxNumberOfLayerPairs * maxNumberOfDoublets * sizeof(GPU::CACell)
    ) == cudaSuccess;

    using QuadrupletVector = GPU::SimpleVector<GPU::Quadruplet>;
    assert(d_foundNtuplets.size() == nStreams);
    assert(device_isOuterHitOfCell.size() == nStreams);
    for (unsigned int i = 0 ; i < nStreams ; ++i) {

        // Allocate memory for d_foundNtuplets data
        GPU::Quadruplet *data_ptr = nullptr;
        success = success && cudaMalloc(
            &data_ptr,
            maxNumberOfQuadruplets * sizeof(GPU::Quadruplet)
        ) == cudaSuccess;

        // Construct the corresponding temporary object (host memory already allocated)
        new (tmp_foundNtuplets[i]) QuadrupletVector(maxNumberOfQuadruplets, data_ptr);
        assert(tmp_foundNtuplets[i]->m_size == 0);
        assert(tmp_foundNtuplets[i]->m_data == data_ptr);
        assert(tmp_foundNtuplets[i]->maxSize == static_cast<int>(maxNumberOfQuadruplets));

        // Allocate memory for the object on the device...
        success = success && cudaMalloc(
            &d_foundNtuplets[i],
            sizeof(QuadrupletVector)
        ) == cudaSuccess
        // ... and copy the object to the device.
        && cudaMemcpy(
            d_foundNtuplets[i],
            tmp_foundNtuplets[i],
            sizeof(QuadrupletVector),
            cudaMemcpyHostToDevice
        ) == cudaSuccess;

        // Allocate device_isOuterHitOfCell
        success = success && cudaMalloc(
            &device_isOuterHitOfCell[i],
            maxNumberOfLayers * maxNumberOfHits * sizeof(GPU::SimpleVector<unsigned int>)
        ) == cudaSuccess;
        // Allocate dense buffer for the underlying data
        unsigned int *denseBuffer = nullptr;
        success = success && cudaMalloc(
            &denseBuffer,
            maxNumberOfLayers * maxNumberOfHits * maxCellsPerHit * sizeof(unsigned int)
        ) == cudaSuccess;
        // Construct objects in-place
        for (unsigned int j = 0 ; j < maxNumberOfLayers ; ++j) {
            for (unsigned int k = 0 ; k < maxNumberOfHits ; ++k) {
                const unsigned int idx = j * maxNumberOfHits + k;
                unsigned int *const data_ptr = denseBuffer + idx * maxCellsPerHit;
                new (&tmp_isOuterHitOfCell[i][idx])
                    GPU::SimpleVector<unsigned int>(maxCellsPerHit, data_ptr);
                assert(tmp_isOuterHitOfCell[i][idx].m_data == data_ptr);
                assert(tmp_isOuterHitOfCell[i][idx].m_size == 0);
                assert(tmp_isOuterHitOfCell[i][idx].maxSize == static_cast<int>(maxCellsPerHit));
            }
        }
        // Initialize data arrays to zero
        success = success && cudaMemset(
            denseBuffer,
            0,
            maxNumberOfLayers * maxNumberOfHits * maxCellsPerHit * sizeof(unsigned int)
        ) == cudaSuccess
        // Copy temporary objects to device
        && cudaMemcpy(
            device_isOuterHitOfCell[i],
            tmp_isOuterHitOfCell[i],
            maxNumberOfLayers * maxNumberOfHits * sizeof(GPU::SimpleVector<unsigned int>),
            cudaMemcpyHostToDevice
        ) == cudaSuccess;
    }

    if (!success) {
        freeDeviceMemory();
    } else {
        assert(d_regionParams);
        assert(d_events);
        assert(d_indices);
        assert(d_doublets);
        assert(d_layers);
        assert(d_x);
        assert(d_y);
        assert(d_z);
        assert(d_rootLayerPairs);
        assert(device_theCells);
        for (unsigned int i = 0 ; i < nStreams ; ++i) {
            assert(device_isOuterHitOfCell[i]);
            assert(d_foundNtuplets[i]);
            for (unsigned int j = 0 ; j < maxNumberOfLayers ; ++j) {
                for (unsigned int k = 0 ; k < maxNumberOfHits ; ++k) {
                    const unsigned int idx = j * maxNumberOfHits + k;
                    assert(tmp_isOuterHitOfCell[i][idx].m_data);
                }
            }
        }
    }

    return success;
}

void CUDACellularAutomaton::freeDeviceMemory()
{
    cudaFree(d_regionParams); d_regionParams = nullptr;
    cudaFree(d_events); d_events = nullptr;
    cudaFree(d_indices); d_indices = nullptr;
    cudaFree(d_doublets); d_doublets = nullptr;
    cudaFree(d_layers); d_layers = nullptr;
    cudaFree(d_x); d_x = nullptr;
    cudaFree(d_y); d_y = nullptr;
    cudaFree(d_z); d_z = nullptr;
    cudaFree(d_rootLayerPairs); d_rootLayerPairs = nullptr;
    cudaFree(device_theCells); device_theCells = nullptr;
    for (unsigned int i = 0 ; i < nStreams ; ++i) {
        cudaMemcpy(
            tmp_foundNtuplets[i],
            d_foundNtuplets[i],
            sizeof(GPU::SimpleVector<GPU::Quadruplet>),
            cudaMemcpyDeviceToHost
        );
        cudaFree(d_foundNtuplets[i]); d_foundNtuplets[i] = nullptr;
        cudaFree(tmp_foundNtuplets[i]->m_data); tmp_foundNtuplets[i]->m_data = nullptr;
        cudaFree(device_isOuterHitOfCell[i]); device_isOuterHitOfCell[i] = nullptr;
        // Recall that the allocation was contiguous
        cudaFree(tmp_isOuterHitOfCell[i][0].m_data);
        // Set all pointers pointing inside the freed buffer to `nullptr`
        for (unsigned int j = 0 ; j < maxNumberOfLayers ; ++j) {
            for (unsigned int k = 0 ; k < maxNumberOfHits ; ++k) {
                const unsigned int idx = j * maxNumberOfHits + k;
                tmp_isOuterHitOfCell[i][idx].m_data = nullptr;
            }
        }
    }
}

bool CUDACellularAutomaton::createCUDAStreams()
{
    assert(streams.size() == 0);
    streams.reserve(nStreams);
    for (unsigned int i = 0 ; i < nStreams ; ++i) {
        // hpx::cerr << "Creating CUDA stream " << i << "... " << hpx::flush;
        streams.push_back(nullptr);
        auto err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            // The last stream is not valid -> don't destroy it
            streams.pop_back();
            // hpx::cerr << failed << hpx::endl << hpx::flush;
            destroyCUDAStreams();
            return false;
        }
        // hpx::cerr << ok << hpx::endl << hpx::flush;
    }
    assert(streams.size() == nStreams);
    // Crete streamInfo
    streamInfo.reserve(nStreams);
    for (unsigned int i = 0 ; i < nStreams ; ++i) {
        streamInfo.push_back(std::make_tuple(this, i, 0));
    }
    assert(streamInfo.size() == nStreams);
    return true;
}

void CUDACellularAutomaton::destroyCUDAStreams()
{
    for (cudaStream_t& str : streams) {
        cudaStreamSynchronize(str);
        cudaStreamDestroy(str);
    }
    streams.clear();
}

void CUDACellularAutomaton::createChannels()
{
    //std::cerr << "Creating stream queue... ";
    assert(streams.size() == nStreams);
    for (unsigned int i = 0 ; i < streams.size() ; ++i) {
        streamQueue.set(i);
    }
    //std::cerr << ok << std::endl;

    //std::cerr << "Creating buffer queue... ";
    for (unsigned int i = 0 ; i < eventQueueSize ; ++i) {
        resourceQueue.set(i);
    }
    //std::cerr << ok << std::endl;

    assert(kernelsDone.size() == nStreams);
}

void CUDACellularAutomaton::destroyChannels()
{
    //std::cerr << "Destroying stream queue... ";
    for (unsigned int i = 0 ; i < nStreams ; ++i) {
        streamQueue.get(hpx::launch::sync);
    }
    //std::cerr << ok << std::endl;

    //std::cerr << "Destroying buffer queue... ";
    for (unsigned int i = 0 ; i < eventQueueSize ; ++i) {
        resourceQueue.get(hpx::launch::sync);
    }
    //std::cerr << ok << std::endl;

    // kernelsDone does not need any specific finalization step
    // Its elements should already be empty
}

std::vector<Host::Quadruplet>
CUDACellularAutomaton::run(
    Host::Event event
)
{
    //std::cerr << "Entering run()" << std::endl;
    // Request pinned buffers (may suspend thread until resources are available)
    //std::cerr << "Requesting buffers... ";
    auto f_bufferIndex = resourceQueue.get();
    const unsigned int bufferIndex = f_bufferIndex.get();
    //std::cerr << ok << "(allocated buffer #" << bufferIndex << ")" << std::endl;

    // Copy event data to pinned memory
    //std::cerr << "Copying events to buffers... ";
    copyEventToPinnedBuffers(event, bufferIndex);
    //std::cerr << ok << std::endl;

    // Request a CUDA stream (may suspend until it is available)
    //std::cerr << "Requesting stream... ";
    auto f_streamIndex = streamQueue.get();
    const unsigned int streamIndex = f_streamIndex.get();
    //std::cerr << ok << "(allocated stream #" << streamIndex << ")" << std::endl;
    cudaSetDevice(gpuIndex);
    debugSynchronizeAndCheck(streams[streamIndex]);

    // Asynchronously copy data to the GPU
    //std::cerr << "Launching asynchronous copy to GPU... ";
    asyncCopyEventToGPU(bufferIndex, streamIndex);
    //std::cerr << ok << std::endl;
    debugSynchronizeAndCheck(streams[streamIndex]);

    // Set stream info (data member passed to callbacks)
    std::get<2>(streamInfo[streamIndex]) = bufferIndex;

    // Define grid and block sizes
    const std::array<unsigned int, 3> blockSize{256, 1, 1};
    const std::array<unsigned int, 3> numberOfBlocks_create{
        32, h_events[bufferIndex].numberOfLayerPairs, 1
    };
    const std::array<unsigned int, 3> numberOfBlocks_connect{
        16, h_events[bufferIndex].numberOfLayerPairs, 1
    };
    const std::array<unsigned int, 3> numberOfBlocks_find{
        8, h_events[bufferIndex].numberOfRootLayerPairs, 1
    };

    // First indices in device memory for current event
    auto d_firstLayerPairInEvt = maxNumberOfLayerPairs * streamIndex;
    auto d_firstLayerInEvt = maxNumberOfLayers * streamIndex;
    auto d_firstRootLayerPairInEvt = maxNumberOfRootLayerPairs * streamIndex;
    auto d_firstDoubletInEvent = d_firstLayerPairInEvt * maxNumberOfDoublets;

    // Launch kernels using wrappers
    // std::cerr << "Kernel: debug_input_data()... ";
    // kernel::debug::input_data(
    //     {1,1,1}, {1,1,1}, 0, streams[streamIndex],
    //     &d_events[streamIndex],
    //     &d_doublets[d_firstLayerPairInEvt],
    //     &d_layers[d_firstLayerInEvt],
    //     d_regionParams,
    //     maxNumberOfHits
    // );
    // std::cerr << ok << std::endl;
    //std::cerr << "Kernel: create()... ";
    kernel::create(
        numberOfBlocks_create, blockSize, 0, streams[streamIndex],
        &d_events[streamIndex],
        &d_doublets[d_firstLayerPairInEvt],
        &d_layers[d_firstLayerInEvt],
        &device_theCells[d_firstDoubletInEvent],
        device_isOuterHitOfCell[streamIndex],
        d_foundNtuplets[streamIndex],
        d_regionParams,
        maxNumberOfDoublets,
        maxNumberOfHits
    );
    debugSynchronizeAndCheck(streams[streamIndex]);
    //std::cerr << ok << std::endl;
    // std::cerr << "Kernel: debug()... ";
    // kernel::debug::all(
    //     {1,1,1}, {1,1,1}, 0, streams[streamIndex],
    //     &d_events[streamIndex],
    //     &d_doublets[d_firstLayerPairInEvt],
    //     &d_layers[d_firstLayerInEvt],
    //     &device_theCells[d_firstDoubletInEvent],
    //     device_isOuterHitOfCell[streamIndex],
    //     d_foundNtuplets[streamIndex],
    //     d_regionParams,
    //     thetaCut, phiCut, hardPtCut,
    //     maxNumberOfDoublets,
    //     maxNumberOfHits
    // );
    // std::cerr << ok << std::endl;

    //std::cerr << "Kernel: connect()... ";
    kernel::connect(
        numberOfBlocks_connect, blockSize, 0, streams[streamIndex],
        &d_events[streamIndex],
        &d_doublets[d_firstLayerPairInEvt],
        &device_theCells[d_firstDoubletInEvent],
        device_isOuterHitOfCell[streamIndex],
        d_regionParams,
        thetaCut,
        phiCut,
        hardPtCut,
        maxNumberOfDoublets,
        maxNumberOfHits
    );
    debugSynchronizeAndCheck(streams[streamIndex]);
    //std::cerr << ok << std::endl;
    // std::cerr << "Kernel: debug_connect()... ";
    // kernel::debug::connect(
    //     {1,1,1}, {1,1,1}, 0, streams[streamIndex],
    //     &d_events[streamIndex],
    //     &d_doublets[d_firstLayerPairInEvt],
    //     &device_theCells[d_firstDoubletInEvent],
    //     device_isOuterHitOfCell[streamIndex],
    //     d_regionParams,
    //     maxNumberOfDoublets,
    //     maxNumberOfHits
    // );
    // std::cerr << ok << std::endl;

    //std::cerr << "Kernel: find_ntuplets()... ";
    constexpr unsigned int minHitsPerNtuplet = 4; // We want quadruplets -> 4
    kernel::find_ntuplets(
        numberOfBlocks_find, blockSize, 0, streams[streamIndex],
        &d_events[streamIndex],
        &d_doublets[d_firstLayerPairInEvt],
        &device_theCells[d_firstDoubletInEvent],
        d_foundNtuplets[streamIndex],
        &d_rootLayerPairs[d_firstRootLayerPairInEvt],
        minHitsPerNtuplet,
        maxNumberOfDoublets
    );
    debugSynchronizeAndCheck(streams[streamIndex]);
    //std::cerr << ok << std::endl;
    // std::cerr << "Kernel: debug_find_ntuplets()... ";
    // kernel::debug::find_ntuplets(
    //     {1,1,1}, {1,1,1}, 0, streams[streamIndex],
    //     &d_events[streamIndex],
    //     &d_doublets[d_firstLayerPairInEvt],
    //     &device_theCells[d_firstDoubletInEvent],
    //     d_foundNtuplets[streamIndex],
    //     &d_rootLayerPairs[d_firstRootLayerPairInEvt],
    //     minHitsPerNtuplet,
    //     maxNumberOfDoublets
    // );
    // std::cerr << ok << std::endl;

    // Asynchronously copy results from GPU to host
    //std::cerr << "Starting asynchronous copy of the results to the host... ";
    asyncCopyResultsToHost(streamIndex, bufferIndex);
    //std::cerr << ok << std::endl;
    debugSynchronizeAndCheck(streams[streamIndex]);

    // Reset the CA by setting the GPU buffer to clean state
    //std::cerr << "Enqueuing asynchronous CA reset... ";
    asyncResetCAState(streamIndex);
    //std::cerr << ok << std::endl;
    debugSynchronizeAndCheck(streams[streamIndex]);

    // Suspend thread and resume it with callback

    // The asynchronous callback will set kernelsDone[streamIndex] when all
    // asynchronous operations have completed.
    enqueueCallback(streamIndex);

    // Request a future representing the asynchronous operations
    auto f_done = kernelsDone[streamIndex].get();

    // Suspend this thread until asynchronous operations have completed
    f_done.get();

    // Give back the buffers and the stream
    // const std::size_t n_found_quad = h_foundNtuplets[bufferIndex]->m_size;
    //std::cerr << "Copying " << n_found_quad << " quadruplets to vector...";
    auto quadruplets = makeQuadrupletVector(bufferIndex);
    //std::cerr << ok << std::endl;
    // hpx::cerr << "Copied " << quadruplets.size() << " quadruplets to vector." << std::endl << std::flush;

    debugSynchronizeAndCheck(streams[streamIndex]);

    //std::cerr << "Relinquishing buffer " << bufferIndex << "... ";
    resourceQueue.set(bufferIndex);
    //std::cerr << ok << std::endl;

    //std::cerr << "Relinquishing stream " << streamIndex << "... ";
    streamQueue.set(streamIndex);
    //std::cerr << ok << std::endl;

    //std::cerr << "Returning NOW!" << std::endl;
    return quadruplets;
}

void CUDACellularAutomaton::copyEventToPinnedBuffers(
    const Host::Event& evt, const unsigned int idx
)
{
    // Copy event
    h_events[idx].eventId = evt.eventId;
    h_events[idx].numberOfRootLayerPairs = 0;
    h_events[idx].numberOfLayers = evt.hitsLayers.size();
    h_events[idx].numberOfLayerPairs = evt.doublets.size();

    // Initialize layer pairs array
    for (unsigned int j = 0 ; j < maxNumberOfLayerPairs ; ++j) {
        unsigned int doubletIdx = idx * maxNumberOfLayerPairs + j;
        h_doublets[doubletIdx].size = 0;
    }

    // Initialize layers array
    for (unsigned int j = 0 ; j < maxNumberOfLayers ; ++j) {
        unsigned int layerIdx = idx * maxNumberOfLayers + j;
        h_layers[layerIdx].size = 0;
    }

    // Copy doublets
    for (unsigned int j = 0 ; j < evt.doublets.size() ; ++j) {

        unsigned int layerPairIndex = idx * maxNumberOfLayerPairs + j;
        h_doublets[layerPairIndex].size = evt.doublets[j].size;
        h_doublets[layerPairIndex].innerLayerId = evt.doublets[j].innerLayerId;
        h_doublets[layerPairIndex].outerLayerId = evt.doublets[j].outerLayerId;

        // Check if inner layer is a root layer
        // Create a root layer pair if it does not already exist
        for (unsigned int l = 0 ; l < evt.rootLayers.size() ; ++l) {
            if (static_cast<unsigned int>(evt.rootLayers[l]) == h_doublets[layerPairIndex].innerLayerId) {
                unsigned int rootLayerPairId = idx * maxNumberOfRootLayerPairs
                    + h_events[idx].numberOfRootLayerPairs;
                h_rootLayerPairs[rootLayerPairId] = j;
                ++h_events[idx].numberOfRootLayerPairs;
            }
        }

        // Copy doublet indices
        for (unsigned int l = 0 ; l < evt.doublets[j].size ; ++l) {
            unsigned int hitId = (layerPairIndex * maxNumberOfDoublets + l) * 2;
            h_indices[hitId] = evt.doublets[j].indices[2*l];
            h_indices[hitId + 1] = evt.doublets[j].indices[2*l + 1];
        }
    }

    // Copy layers
    for (unsigned int j = 0 ; j < evt.hitsLayers.size() ; ++j) {

        unsigned int layerIdx = idx * maxNumberOfLayers + j;
        h_layers[layerIdx].size = evt.hitsLayers[j].size;
        h_layers[layerIdx].layerId = evt.hitsLayers[j].layerId;

        // Copy layer hits
        for (unsigned int l = 0 ; l < evt.hitsLayers[j].size ; ++l) {
            unsigned int hitId = layerIdx * maxNumberOfHits + l;
            h_x[hitId] = evt.hitsLayers[j].x[l];
            h_y[hitId] = evt.hitsLayers[j].y[l];
            h_z[hitId] = evt.hitsLayers[j].z[l];
        }
    }
}

void CUDACellularAutomaton::asyncCopyEventToGPU(
    unsigned int i, unsigned int streamIndex
)
{
    // First indices in device memory for current event
    auto d_firstLayerPairInEvt = maxNumberOfLayerPairs * streamIndex;
    auto d_firstLayerInEvt = maxNumberOfLayers * streamIndex;
    auto d_firstDoubletInEvent = d_firstLayerPairInEvt * maxNumberOfDoublets;
    auto d_firstHitInEvent = d_firstLayerInEvt * maxNumberOfHits;

    // First indices in pinned host memory for current event
    auto h_firstLayerPairInEvt = maxNumberOfLayerPairs * i;
    auto h_firstLayerInEvt = maxNumberOfLayers * i;
    auto h_firstDoubletInEvent = h_firstLayerPairInEvt * maxNumberOfDoublets;
    auto h_firstHitInEvent = h_firstLayerInEvt * maxNumberOfHits;

    // Copy doublet indices
    for (unsigned int j = 0 ; j < h_events[i].numberOfLayerPairs ; ++j)
    {
        tmp_layerDoublets[streamIndex][j] = h_doublets[h_firstLayerPairInEvt + j];
        tmp_layerDoublets[streamIndex][j].indices =
            &d_indices[(d_firstDoubletInEvent + j * maxNumberOfDoublets) * 2];

        cudaMemcpyAsync(
            &d_indices[(d_firstDoubletInEvent + j * maxNumberOfDoublets) * 2],
            &h_indices[(h_firstDoubletInEvent + j * maxNumberOfDoublets) * 2],
            tmp_layerDoublets[streamIndex][j].size * 2 * sizeof(unsigned int),
            cudaMemcpyHostToDevice, streams[streamIndex]
        );
    }

    // Copy layer hit positions
    for (unsigned int j = 0 ; j < h_events[i].numberOfLayers ; ++j)
    {
        tmp_layers[streamIndex][j] = h_layers[h_firstLayerInEvt + j];
        tmp_layers[streamIndex][j].x = &d_x[d_firstHitInEvent + maxNumberOfHits * j];

        cudaMemcpyAsync(
            &d_x[d_firstHitInEvent + maxNumberOfHits * j],
            &h_x[h_firstHitInEvent + j * maxNumberOfHits],
            tmp_layers[streamIndex][j].size * sizeof(float),
            cudaMemcpyHostToDevice, streams[streamIndex]
        );

        tmp_layers[streamIndex][j].y = &d_y[d_firstHitInEvent + maxNumberOfHits * j];
        cudaMemcpyAsync(
            &d_y[d_firstHitInEvent + maxNumberOfHits * j],
            &h_y[h_firstHitInEvent + j * maxNumberOfHits],
            tmp_layers[streamIndex][j].size * sizeof(float),
            cudaMemcpyHostToDevice, streams[streamIndex]
        );

        tmp_layers[streamIndex][j].z = &d_z[d_firstHitInEvent + maxNumberOfHits * j];
        cudaMemcpyAsync(
            &d_z[d_firstHitInEvent + maxNumberOfHits * j],
            &h_z[h_firstHitInEvent + j * maxNumberOfHits],
            tmp_layers[streamIndex][j].size * sizeof(float),
            cudaMemcpyHostToDevice, streams[streamIndex]
        );
    }

    // Copy layers, root layers and layer pairs
    cudaMemcpyAsync(
        &d_rootLayerPairs[maxNumberOfRootLayerPairs * streamIndex],
        &h_rootLayerPairs[maxNumberOfRootLayerPairs * i],
        h_events[i].numberOfRootLayerPairs * sizeof(unsigned int),
        cudaMemcpyHostToDevice, streams[streamIndex]
    );

    cudaMemcpyAsync(
        &d_doublets[d_firstLayerPairInEvt],
        tmp_layerDoublets[streamIndex],
        h_events[i].numberOfLayerPairs * sizeof(GPU::LayerDoublets),
        cudaMemcpyHostToDevice, streams[streamIndex]
    );

    cudaMemcpyAsync(
        &d_layers[d_firstLayerInEvt],
        tmp_layers[streamIndex],
        h_events[i].numberOfLayers * sizeof(GPU::LayerHits),
        cudaMemcpyHostToDevice, streams[streamIndex]
    );

    // Copy event information
    cudaMemcpyAsync(
        &d_events[streamIndex],
        &h_events[i],
        sizeof(GPU::Event),
        cudaMemcpyHostToDevice, streams[streamIndex]
    );
}

void CUDACellularAutomaton::asyncCopyResultsToHost(
    unsigned int streamIndex, unsigned int bufferIndex
)
{
    // Copy object
    cudaMemcpyAsync(
        tmp_foundNtuplets[streamIndex],
        d_foundNtuplets[streamIndex],
        sizeof(GPU::SimpleVector<GPU::Quadruplet>),
        cudaMemcpyDeviceToHost,
        streams[streamIndex]
    );

    // Copy quadruplet data
    cudaMemcpyAsync(
        h_foundNtuplets[bufferIndex]->m_data,
        tmp_foundNtuplets[streamIndex]->m_data,
        tmp_foundNtuplets[streamIndex]->maxSize * sizeof(GPU::Quadruplet),
        cudaMemcpyDeviceToHost,
        streams[streamIndex]
    );

    cudaStreamAddCallback(
        streams[streamIndex],
        [](cudaStream_t str, cudaError_t status, void *data) -> void {
            // std::cerr << "Executing callback... ";
            auto tup = *static_cast<std::tuple<CUDACellularAutomaton*, unsigned int, unsigned int>*>(data);
            auto this_ = std::get<0>(tup);
            auto streamIndex = std::get<1>(tup);
            auto bufferIndex = std::get<2>(tup);
            this_->h_foundNtuplets[bufferIndex]->m_size = this_->tmp_foundNtuplets[streamIndex]->m_size;
            assert(this_->h_foundNtuplets[bufferIndex]->maxSize
                   >= this_->tmp_foundNtuplets[streamIndex]->m_size);
            // std::cerr << ok << std::endl;
        },
        static_cast<void*>(&streamInfo[streamIndex]),
        0
    );
}

void CUDACellularAutomaton::asyncResetCAState(
    unsigned int streamIndex
)
{
    // Recall that the allocation was dense
    // So we can reset everything in a single call to `cudaMemsetAsync`
    cudaMemsetAsync(
        tmp_isOuterHitOfCell[streamIndex][0].m_data,
        0,
        maxNumberOfLayers * maxNumberOfHits * maxCellsPerHit * sizeof(unsigned int),
        streams[streamIndex]
    );

    assert(tmp_isOuterHitOfCell[streamIndex][0].m_size == 0);
    assert(tmp_isOuterHitOfCell[streamIndex][0].maxSize == static_cast<int>(maxCellsPerHit));

    cudaMemcpyAsync(
        device_isOuterHitOfCell[streamIndex],
        tmp_isOuterHitOfCell[streamIndex],
        maxNumberOfLayers * maxNumberOfHits * sizeof(GPU::SimpleVector<unsigned int>),
        cudaMemcpyHostToDevice,
        streams[streamIndex]
    );
}

std::vector<Host::Quadruplet> CUDACellularAutomaton::makeQuadrupletVector(
    unsigned int bufferIndex
) const
{
    assert(h_foundNtuplets[bufferIndex]->m_size >= 0);
    const auto sz = static_cast<std::size_t>(h_foundNtuplets[bufferIndex]->m_size);
    std::vector<Host::Quadruplet> quadruplets;
    quadruplets.reserve(sz);
    for (std::size_t i = 0 ; i < sz ; ++i) {
        auto const &q_data = h_foundNtuplets[bufferIndex]->m_data[i].layerPairsAndCellId;
        Host::Quadruplet quad{{
            {q_data[0].x, q_data[0].y},
            {q_data[1].x, q_data[1].y},
            {q_data[2].x, q_data[2].y}
        }};
        quadruplets.push_back(quad);
    }
    return quadruplets;
}

void CUDACellularAutomaton::enqueueCallback(unsigned int streamIndex)
{
    cudaStreamAddCallback(
        streams[streamIndex],
        [](cudaStream_t str, cudaError_t status, void *data) -> void {
            auto done = static_cast<hpx::lcos::local::one_element_channel<void>*>(data);
            done->set();
        },
        static_cast<void*>(&kernelsDone[streamIndex]),
        0
    );
}
