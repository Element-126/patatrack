#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/lcos/local/spinlock.hpp>

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
    assert(batchSize > 0);
    const std::size_t sendersPerCA = vm["senders-per-ca"].as<std::size_t>();

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
    std::vector<hpx::future<QuadrupletVector>> f_allQuadruplets(nTotalEvents);
    std::vector<std::size_t> nFoundQuadruplets(nTotalEvents, 0);

    // Time measurement
    std::chrono::high_resolution_clock clock;
    auto const global_start_time = clock.now();

    std::size_t idx(0);
    using mutex_t = hpx::lcos::local::spinlock;
    mutex_t mtx_idx;
    hpx::parallel::static_chunk_size onePerThread(1);
    hpx::parallel::for_loop(hpx::parallel::par.with(onePerThread), 0, nGPUs * sendersPerCA,
        [&idx, &mtx_idx, &f_allQuadruplets, &nFoundQuadruplets, nTotalEvents,
         &events, nEvents, &cellularAutomatons, batchSize]
        (std::size_t caIndex) -> void
    {
        // Select CA
        auto &myCA = cellularAutomatons[caIndex % cellularAutomatons.size()];
        CUDACellularAutomaton_run_action ca_action;

        // Loop until all batches have been processed
        bool finished = false;
        std::size_t rg_start(0), rg_end(0);
        while (!finished) {

            // Asynchronously retreive the results from the previous batch
            assert(rg_start <= rg_end);
            assert(rg_end <= nTotalEvents);
            hpx::future<void> f_done = hpx::async(
                [rg_start, rg_end, &f_allQuadruplets, &nFoundQuadruplets]() -> void {

                for (std::size_t i = rg_start ; i < rg_end ; ++i) {
                    nFoundQuadruplets[i] = f_allQuadruplets[i].get().size();
                }
            });

            // Increase the index to take ownership of a range of events
            {
                std::lock_guard<mutex_t> l(mtx_idx);
                if (idx >= nTotalEvents) {
                    finished = true;
                    rg_start = rg_end = nTotalEvents;
                    assert(idx == nTotalEvents);
                } else {
                    rg_start = idx;
                    rg_end = rg_start + batchSize;
                    // rg_end = rg_start + (batchSize - 1) * (nTotalEvents - rg_start) / nTotalEvents + 1;
                    rg_end = std::min(rg_end, nTotalEvents);
                    idx = rg_end;
                }
            } // mtx_idx unlocked
#if defined(DEBUG)
            if (finished) {
                hpx::cerr << "CA" << caIndex << " finished" << hpx::endl << hpx::flush;
            } else {
                hpx::cerr << "CA" << caIndex << ": " << rg_start << " -- " << rg_end << hpx::endl << hpx::flush;
            }
#endif // DEBUG

            // Send events to CA worker
            assert(rg_start <= rg_end);
            assert(rg_end <= nTotalEvents);
            for (std::size_t i = rg_start ; i < rg_end ; ++i) {
                assert(!finished);
                const auto &evt = events[i % nEvents];
                f_allQuadruplets[i] = hpx::async(ca_action, myCA, evt);
            }

            // Synchronize
            f_done.get();
        }
    });

    auto const global_end_time = clock.now();
    const double eventProcessingTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(global_end_time - global_start_time).count() / 1000.;

    hpx::cout << hpx::flush;
    // hpx::cout << "All received batches:" << std::endl << hpx::flush;
    // std::size_t sum = 0;
    // for (std::size_t idx = 0 ; idx < nTotalEvents ; ++idx) {
    //     sum += nFoundQuadruplets[idx];
    //     const std::size_t n = idx % nEvents;
    //     hpx::cout << "    #" << n << ":\t" << nFoundQuadruplets[idx] << "\t" << sum << std::endl;
    //     if (idx % nEvents == nEvents - 1) {
    //         const std::size_t k = idx / nEvents;
    //         hpx::cout << "Iteration #" << k << ": sum = " << sum << std::endl << hpx::flush;
    //         sum = 0;
    //     }
    // }
    hpx::cout << "Processed " << nTotalEvents << " events" << hpx::endl << hpx::flush;
    hpx::cerr << "Overall event processing rate: " << nTotalEvents / eventProcessingTime << " Hz" << hpx::endl << hpx::flush;

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
        ("senders-per-ca,m",
         po::value<std::size_t>()->default_value(1),
         "Number of threads sending events to each CA worker");

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
