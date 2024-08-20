#!/bin/bash

#PBS -N PREPARE
#PBS -l mem=80GB

cd ${PBS_O_WORKDIR}
apptainer run image.sif estimate.R
