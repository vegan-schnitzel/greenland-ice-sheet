#!/usr/bin/env bash

# extract & and merge variables from ERA5 data &
# doing some pre-processing

CDO=$(which cdo)
CDO_FLGS="-L -O -f nc4"

DATAPATH=/daten/reana/arch/reanalysis/reanalysis/DKRZ/IFS/ERA5/day/atmos

# I could have also written this as a loop...

# input temperature (2m) and precipitation field
INPUTPR=$(ls ${DATAPATH}/pr/r1i1p1/pr_day_reanalysis_era5_r1i1p1_201?0101-201?1231.nc)
INPUTTAS=$(ls ${DATAPATH}/tas/r1i1p1/tas_day_reanalysis_era5_r1i1p1_201?0101-201?1231.nc)

# get & merge temperature
$CDO $CDO_FLGS -ydaymean -del29feb -sellonlatbox,288,349,59,85 \
    -mergetime -selname,tas $INPUTTAS tas-greenland-merged.nc
# standard deviation of mean
$CDO $CDO_FLGS -ydaystd -del29feb -sellonlatbox,288,349,59,85 \
    -mergetime -selname,tas $INPUTTAS tas-greenland-merged-std.nc   
# get & merge precipitation
$CDO $CDO_FLGS -ydaymean -del29feb -sellonlatbox,288,349,59,85 \
    -mergetime -selname,pr $INPUTPR pr-greenland-merged.nc

