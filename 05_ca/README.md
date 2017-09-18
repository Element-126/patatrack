Compile with:

    $ make -j 2
    
To rebuild everything, add the `-B` option:

    $ make -j 2 -B

Execute with: 

    $ ./ca -n 100 --iterations=10000 --streams=8 --queue-size=8 --batch-size=1000 -i <path-to-data>/parsed_ttbar50PU_1000evts.txt --max-cells-per-hit=150 --gpus=0,1,2,3,4,5,6,7

You may need to add the `build/` directory, where `libkernels.so` is located, to your `LD_LIBRARY_PATH`.

To see all available command-line parameters, use `./ca --help`.
