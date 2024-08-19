#!/bin/bash

#PBS -N PREPARE
#PBS -l mem=64GB

cd ${PBS_O_WORKDIR}
apptainer run image.sif estimate.R
