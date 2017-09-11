#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/program_options.hpp>

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

    const Host::MaxHitsAndLayers maxHitsAndLayers(
        max_hits,
        max_doublets,
        maxNumberOfLayers,
        maxNumberOfLayerPairs,
        maxNumberOfRootLayerPairs
    );

    // // Dispatch events to the various localities, in a round-robin fashion
    // auto localities = hpx::find_all_localities();
    // const std::size_t n_localities = localities.size();
    // std::cout << "Processing " << nEvents << " events on "
    //           << n_localities << " localities" << std::endl;
    // std::vector<hpx::future<std::vector<std::array<std::array<int, 2>, 3>>>> quadruplets_fut(events.size());
    // for (std::size_t i = 0 ; i < nEvents ; ++i) {
    //     auto loc = localities[i % n_localities];
    //     assert(i < events.size());
    //     quadruplets_fut[i] = hpx::async(process_event_action(), loc, events[i]);
    // }

    // // Wait for everyone to finish
    // hpx::wait_all(quadruplets_fut);
    // std::cout << "Done" << std::endl;

    constexpr std::size_t gpuIndex = 0;
    hpx::cout << "main(): Remotely instantiating CA... ";
    auto f_ca = hpx::new_<CUDACellularAutomaton>(
        hpx::find_here(),
        maxNumberOfQuadruplets,
        maxCellsPerHit,
        region,
        cuts,
        maxHitsAndLayers,
        gpuIndex,
        nStreams,
        eventQueueSize
    );

    hpx::cout << "synchronizing... ";
    auto ca = f_ca.get();
    hpx::cout << "\033[1;32mOK\033[0m" << std::endl;

    hpx::cout << "main(): Sending events..." << std::endl;
    using QuadrupletVector = std::vector<Host::Quadruplet>;
    CUDACellularAutomaton_run_action ca_action;
    std::vector<hpx::future<QuadrupletVector>> f_allQuadruplets(nTotalEvents);
    std::vector<std::size_t> nFoundQuadruplets(nTotalEvents, 0);
    std::chrono::high_resolution_clock clock;
    const auto start_time = clock.now();
    for (std::size_t k = 0 ; k < nRepeat ; ++k) {
        for (std::size_t n = 0 ; n < nEvents ; ++n) {
            const auto idx = k*nEvents + n;
            f_allQuadruplets[idx] = hpx::async(ca_action, ca, events[n]);
            // auto fut = hpx::async(ca_action, ca, events[n]);
            // auto fut = hpx::async(ca_action, ca, events[n]);
            // auto quadruplets = fut.get();
            // nFoundQuadruplets[n] = quadruplets.size();
            // std::cerr << "#" << n << ": Found " << quadruplets.size() << " quadruplets:" << std::endl;
            // for (std::size_t i = 0 ; i < quadruplets.size() ; ++i) {
            //     std::cerr << "    " << quadruplets[i] << std::endl;
            // }
        }
    }

    // DEBUG
    // std::size_t nQuad = 0;
    // std::size_t it = 0;
    // do {
    //     std::cerr << it << ":";
    //     auto fut = hpx::async(ca_action, ca, events[11]);
    //     auto quadruplets = fut.get();
    //     nQuad = quadruplets.size();
    //     std::cerr << " found " << nQuad << " quadruplets" << std::endl;
    //     ++it;
    // } while (true);

    // DEBUG
    // std::size_t receivedEvents = 0;
    // while (receivedEvents < nEvents) {
    //     int idx = -1;
    //     for (std::size_t i = 0 ; i < f_allQuadruplets.size() ; ++i) {
    //         if (f_allQuadruplets[i].is_ready()) {
    //             idx = i;
    //             break;
    //         }
    //     }
    //     if (idx >= 0) {
    //         nFoundQuadruplets[idx] = f_allQuadruplets[idx].get().size();
    //         hpx::cout << "main(): Received " << nFoundQuadruplets[idx] << " quadruplets (event #" << idx << ")" << std::endl;
    //         ++receivedEvents;
    //     }
    // }

    // DEBUG: wait for futures in-order
    for (std::size_t k = 0 ; k < nRepeat ; ++k) {
        for (std::size_t n = 0 ; n < nEvents ; ++n) {
            const auto idx = k*nEvents + n;
            nFoundQuadruplets[idx] = f_allQuadruplets[idx].get().size();
            // hpx::cout << "main(): Received " << nFoundQuadruplets[n] << " quadruplets (event #" << n << ")" << std::endl;
        }
    }
    const auto end_time = clock.now();
    const double seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
        / 1000.;

    hpx::cout << "All received batches:" << std::endl;
    for (std::size_t k = 0 ; k < nRepeat ; ++k) {
        for (std::size_t n = 0 ; n < nEvents ; ++n) {
            const auto idx = k*nEvents + n;
            hpx::cout << "    #" << n << ":\t" << nFoundQuadruplets[idx] << std::endl;
        }
    }
    const auto throughput_Hz = nTotalEvents / seconds;
    std::cerr << "Total event processing rate: " << throughput_Hz << " Hz" << std::endl;

    // Finalize HPX runtime
    hpx::cout << "main(): Finalizing HPX NOW!" << std::endl;
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
         "Number of times each event will be processed");

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
