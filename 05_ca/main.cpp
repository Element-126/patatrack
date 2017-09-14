#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/parallel/executors/guided_chunk_size.hpp>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#include <string>
#include <vector>
#include <cassert>
#include <chrono>
#include <fstream>

#include "Event/HostEvent.hpp"
#include "Event/HostRegion.hpp"
#include "parser.hpp"
#include "CellularAutomaton/CUDACellularAutomaton.hpp"

namespace po = boost::program_options;

int hpx_main(po::variables_map& vm)
{
    // Constants
    constexpr std::size_t maxNumberOfLayers = 10;
    constexpr std::size_t maxNumberOfLayerPairs = 13;
    constexpr std::size_t maxNumberOfRootLayerPairs = 3;

    // Read parameters
    const std::size_t max_events = vm["events"].as<std::size_t>();
    const std::string input_file = vm["input-file"].as<std::string>();
    const std::size_t nStreams = vm["streams"].as<std::size_t>();
    const std::size_t eventQueueSize = vm["queue-size"].as<std::size_t>();
    const std::size_t maxNumberOfQuadruplets = vm["max-number-of-quadruplets"].as<std::size_t>();
    const std::size_t maxCellsPerHit = vm["max-cells-per-hit"].as<std::size_t>();
    const Host::Cuts cuts(
        vm["theta-cut"].as<double>(),
        vm["phi-cut"].as<double>(),
        vm["hard-pt-cut"].as<double>()
    );
    const std::size_t nRepeat = vm["iterations"].as<std::size_t>();
    const std::string gpuListString = vm["gpus"].as<std::string>();
    std::vector<std::string> subStrings;
    std::vector<std::size_t> gpuIndices;
    boost::split(subStrings, gpuListString, boost::is_any_of(","));
    std::transform(subStrings.begin(), subStrings.end(), std::back_inserter(gpuIndices),
                   [](std::string const& str) -> std::size_t { return std::stoi(str); });
    const std::size_t nGPUs = gpuIndices.size();
    const std::size_t batchSize = vm["batch-size"].as<std::size_t>();
    const std::size_t chunkSize = vm["chunk-size"].as<std::size_t>();

    // Parse input file
    std::vector<Host::Event> events;
    if (max_events > 0) {
        events.reserve(max_events);
    }
    Host::Region region;
    unsigned int max_hits = 0;
    unsigned int max_doublets = 0;
    parseinputFile(input_file, events, region, max_events, max_hits, max_doublets);
    std::size_t nEvents = events.size();
    const std::size_t nTotalEvents = nEvents * nRepeat;

    std::cerr << "Max number of hits: " << max_hits << std::endl;
    std::cerr << "Max number of doublets: " << max_doublets << std::endl;

    const Host::MaxHitsAndLayers maxHitsAndLayers(
        max_hits,
        max_doublets,
        maxNumberOfLayers,
        maxNumberOfLayerPairs,
        maxNumberOfRootLayerPairs
    );

    auto const localities = hpx::find_all_localities();
    std::vector<hpx::id_type> cellularAutomatons(nGPUs);
    std::vector<hpx::future<hpx::id_type>> f_ca(nGPUs);
    for (std::size_t i = 0 ; i < nGPUs ; ++i) {
        assert(gpuIndices.size() == nGPUs);
        f_ca[i] = hpx::new_<CUDACellularAutomaton>(
            localities[i % localities.size()],
            maxNumberOfQuadruplets,
            maxCellsPerHit,
            region,
            cuts,
            maxHitsAndLayers,
            gpuIndices[i],
            nStreams,
            eventQueueSize
        );
    }

    for (std::size_t i = 0 ; i < nGPUs ; ++i) {
        cellularAutomatons[i] = f_ca[i].get();
    }

    using QuadrupletVector = std::vector<Host::Quadruplet>;
    CUDACellularAutomaton_run_action ca_action;
    std::vector<hpx::future<QuadrupletVector>> f_allQuadruplets(nTotalEvents);
    std::vector<std::size_t> nFoundQuadruplets(nTotalEvents, 0);

    // Fixed chunk size
    // hpx::parallel::static_chunk_size chunkSizeExe(chunkSize);
    // Automatic, but with a minimum chunk size
    hpx::parallel::guided_chunk_size chunkSizeExe(chunkSize);

    // Time measurement
    std::chrono::high_resolution_clock clock;
    double futureProductionTime = 0.;
    double futureRetrievalTime = 0.;
    std::vector<double> timings(batchSize, 0);
    std::vector<std::size_t> counts(batchSize, 0);
    auto const global_start_time = clock.now();

    std::size_t idx = 0, lastIdx = 0;
    while (lastIdx < nTotalEvents)
    {
        // Wait for the results of the previous batch, in-order, in another thread
        hpx::future<double> f_done = hpx::async([idx, lastIdx, &f_allQuadruplets, &nFoundQuadruplets, &clock, &timings, &counts]() {
            auto const start_time = clock.now();
            auto partial_time = clock.now();
            const std::size_t batchSize = timings.size();
            for (std::size_t i = lastIdx ; i < idx ; ++i) {
                auto const last = partial_time;
                nFoundQuadruplets[i] = f_allQuadruplets[i].get().size();
                partial_time = clock.now();
                const std::size_t j = i % batchSize;
                timings[j] += std::chrono::duration_cast<std::chrono::milliseconds>(partial_time - last).count();
                ++counts[j];
            }
            auto const end_time = clock.now();
            return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.;
        });

        // Send the next batch of futures
        const std::size_t nextBatchIdx = std::min(idx + batchSize, nTotalEvents);
        auto const start_time = clock.now();
        hpx::parallel::for_loop(hpx::parallel::par.with(chunkSizeExe), idx, nextBatchIdx, [&](std::size_t i) {
        // for (std::size_t i = idx ; i < nextBatchIdx ; ++i) {
            const auto &ca = cellularAutomatons[i % nGPUs];
            const auto &evt = events[i % nEvents];
            f_allQuadruplets[i] = hpx::async(ca_action, ca, evt);
        });
        // }
        auto const end_time = clock.now();
        futureProductionTime +=
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.;

        futureRetrievalTime += f_done.get(); // Synchronize
        lastIdx = idx;
        idx = nextBatchIdx;
    }
    auto const global_end_time = clock.now();
    const double eventProcessingTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(global_end_time - global_start_time).count() / 1000.;

    for (std::size_t i = 0 ; i < batchSize ; ++i) {
        timings[i] /= static_cast<double>(counts[i]);
    }

    hpx::cout << hpx::flush;
    hpx::cout << "All received batches:" << std::endl << hpx::flush;
    std::size_t sum = 0;
    for (std::size_t idx = 0 ; idx < nTotalEvents ; ++idx) {
        sum += nFoundQuadruplets[idx];
        const std::size_t n = idx % nEvents;
        hpx::cout << "    #" << n << ":\t" << nFoundQuadruplets[idx] << "\t" << sum << std::endl;
        if (idx % nEvents == nEvents - 1) {
            const std::size_t k = idx / nEvents;
            hpx::cout << "Iteration #" << k << ": sum = " << sum << std::endl << hpx::flush;
            sum = 0;
        }
    }
    hpx::cout << "Processed " << nTotalEvents << " events" << hpx::endl << hpx::flush;
    hpx::cerr << "Future production rate: " << nTotalEvents / futureProductionTime << " Hz" << hpx::endl << hpx::flush;
    hpx::cerr << "Future retrieval rate: " << nTotalEvents / futureRetrievalTime << " Hz" << hpx::endl << hpx::flush;
    hpx::cerr << "Overall event processing rate: " << nTotalEvents / eventProcessingTime << " Hz" << hpx::endl << hpx::flush;

    {
        std::ofstream timings_file("timings.dat");
        assert(timings.size() > 0);
        for (auto const &timing: timings) {
            timings_file << timing << std::endl;
        }
    }

    // Finalize HPX runtime
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Specify command-line parameters
    po::options_description desc("Usage: " + std::string(argv[0]) + " options");
    desc.add_options()
        ("help,h", "Display this message")
        ("events,n",
         po::value<std::size_t>()->default_value(0),
         "Specify the number of events to process [default: all]")
        ("input-file,i",
         po::value<std::string>()->default_value("../input/parsed.out"),
         "Specify the path of the input ASCII file "
         "containing the events to process")
        ("streams,s",
         po::value<std::size_t>()->default_value(5),
         "Number of CUDA streams")
        ("queue-size,b",
         po::value<std::size_t>()->default_value(10),
         "Event queue size")
        ("max-number-of-quadruplets",
         po::value<std::size_t>()->default_value(3000),
         "Maximum number of quadruplets")
        ("max-cells-per-hit",
         po::value<std::size_t>()->default_value(100),
         "Maximum number of cells per hit")
        ("theta-cut",
         po::value<double>()->default_value(0.002),
         "θ cut")
        ("phi-cut",
         po::value<double>()->default_value(0.2),
         "φ cut")
        ("hard-pt-cut",
         po::value<double>()->default_value(0.),
         "Hard pT cut")
        ("iterations",
         po::value<std::size_t>()->default_value(1),
         "Number of times each event will be processed")
        ("gpus",
         po::value<std::string>()->default_value("0"),
         "Comma-separated list of GPUs to use")
        ("batch-size,z",
         po::value<std::size_t>()->default_value(100),
         "Number of futures scheduled simultaneously")
        ("chunk-size,k",
         po::value<std::size_t>()->default_value(10),
         "Chunk size for parallel future production");

    po::variables_map vm;
    po::store(
        po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(),
        vm
    );
    po::notify(vm);

    // Display usage if -h/--help was passed
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    // Start HPX runtime
    return hpx::init(desc, argc, argv);
}
