#!/bin/bash

echo "###################################################" >> std.out
echo "###################################################" >> std.out

######### Allocate -> vmap to 100^3, numcells to 150^3
for (( Nx=100; Nx<=110; Nx++ )); do
    if (( Nx % 2 == 0 )); then
        echo "Run with Nx = $Nx"
        .venv/bin/python neighbors_search/scaling.py --Nx=$Nx --mode=allocate --nl-backend=jaxmd_vmap >> std.out 2> std.err
    fi
done

echo "###################################################" >> std.out

for (( Nx=150; Nx<=160; Nx++ )); do
    if (( Nx % 2 == 0 )); then
        echo "Run with Nx = $Nx"
        .venv/bin/python neighbors_search/scaling.py --Nx=$Nx --mode=allocate --nl-backend=jaxmd_scan --num-partitions=4 >> std.out 2> std.err
    fi
done

echo "###################################################" >> std.out
echo "###################################################" >> std.out

######### Update -> vmap to 100^3, numcells to 150^3
for (( Nx=100; Nx<=110; Nx++ )); do
    if (( Nx % 2 == 0 )); then
        echo "Run with Nx = $Nx"
        .venv/bin/python neighbors_search/scaling.py --Nx=$Nx --mode=update --nl-backend=jaxmd_vmap >> std.out 2> std.err
    fi
done

echo "###################################################" >> std.out

# Run a for loop over different Nx values
for (( Nx=150; Nx<=160; Nx++ )); do
    if (( Nx % 2 == 0 )); then
        echo "Run with Nx = $Nx"
        .venv/bin/python neighbors_search/scaling.py --Nx=$Nx --mode=update --nl-backend=jaxmd_scan --num-partitions=4 >> std.out 2> std.err
    fi
done
