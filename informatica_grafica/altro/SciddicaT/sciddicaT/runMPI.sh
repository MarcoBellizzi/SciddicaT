#!/bin/bash
srun mpirun -np 2 ./monolithicMPI ../data/tessina_header.txt ../data/tessina_dem.txt ../data/tessina_source.txt ./tessina_output_serial 4000 && md5sum ./tessina_output_serial && cat ../data/tessina_header.txt ./tessina_output_serial > ./tessina_output_serial.qgis && rm ./tessina_output_serial
