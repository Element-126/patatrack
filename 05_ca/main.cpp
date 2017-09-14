#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#include <string>
#include <vector>
#include <cassert>
#include <chrono>

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
    std::chrono::high_resolution_clock clock;
    hpx::parallel::static_chunk_size chunk_size(1);
    const auto start_time = clock.now();
    hpx::parallel::for_loop(hpx::parallel::execution::par.with(chunk_size), 0, nRepeat, [&](std::size_t k) {
    // for (std::size_t k = 0 ; k < nRepeat ; ++k) {
        for (std::size_t n = 0 ; n < nEvents ; ++n) {
            const auto idx = k*nEvents + n;
            const auto &ca = cellularAutomatons[idx % nGPUs];
            f_allQuadruplets[idx] = hpx::async(ca_action, ca, events[n]);
        }
    });
    // }
    const auto send_future_time = clock.now();
    const double sf_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(send_future_time - start_time).count()
        / 1000.;
    hpx::cout << "Future production rate: " << nTotalEvents / sf_seconds << " Hz" << hpx::endl << hpx::flush;

    // DEBUG: wait for futures in-order
    for (std::size_t k = 0 ; k < nRepeat ; ++k) {
        for (std::size_t n = 0 ; n < nEvents ; ++n) {
            const auto idx = k*nEvents + n;
            auto quadruplets = f_allQuadruplets[idx].get();
            nFoundQuadruplets[idx] = quadruplets.size();
            // std::cerr << "#" << n << ": Found " << quadruplets.size() << " quadruplets:" << std::endl;
            // for (std::size_t i = 0 ; i < quadruplets.size() ; ++i) {
            //     std::cerr << "    " << quadruplets[i] << std::endl;
            // }
        }
    }
    const auto end_time = clock.now();
    const double seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
        / 1000.;

    hpx::cout << hpx::flush;
    // hpx::cout << "All received batches:" << std::endl << hpx::flush;
    // std::size_t sum = 0;
    // for (std::size_t k = 0 ; k < nRepeat ; ++k) {
    //     for (std::size_t n = 0 ; n < nEvents ; ++n) {
    //         const auto idx = k*nEvents + n;
    //         sum += nFoundQuadruplets[idx];
    //         hpx::cout << "    #" << n << ":\t" << nFoundQuadruplets[idx] << "\t" << sum << std::endl;
    //     }
    //     hpx::cout << "Iteration #" << k << ": sum = " << sum << std::endl << hpx::flush;
    //     sum = 0;
    // }
    const auto throughput_Hz = nTotalEvents / seconds;
    hpx::cout << "Processed " << nTotalEvents << " events" << hpx::endl << hpx::flush;
    hpx::cout << "Total event processing rate: " << throughput_Hz << " Hz" << hpx::endl << hpx::flush;

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
         "Comma-separated list of GPUs to use");

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
