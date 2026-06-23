#!/bin/bash
overwrite=true
directory="/backup_fs/Archive/kl909/chapter_1/tides/hdf5/"
ncore=1
# these need to match the filelist below
# this is not a clever script, so easy to get this wrong
startTime=0
dt=900

resolution=500
projection=EPSG:32756
maskfile="../modern_mask.shp"


function join { local IFS="$1"; shift; echo "$*"; }

function process_file {

    processing_file=${1}
    NUMBER=${processing_file%*.h5}  # retain the part before the h5 extension
    NUMBER=${NUMBER##*_}  # retain the part after the last _, which is the number
    echo $NUMBER

    # loop over variables
    file=${processing_file}
    # loop over variables with counter
    python h5_2_raster.py --resolution ${resolution} --wd_mask "${directory}../bathymetry.h5" --velocity --mesh improved_modern.msh -- ${file} temp 
    gdalwarp -cutline ${maskfile} -s_srs ${projection} -of netCDF -r bilinear  -dstnodata -9999 -overwrite temp_u_${NUMBER}.xyz "${file}_u.nc"
    gdalwarp -cutline ${maskfile} -s_srs ${projection} -of netCDF -r bilinear  -dstnodata -9999 -overwrite temp_v_${NUMBER}.xyz "${file}_v.nc"
    rm temp_u_${NUMBER}.xyz
    rm temp_v_${NUMBER}.xyz

}

# loop through files twice; once to get the count and create the times
# then to process files, which is done in parallel.
# temp hack to only do 14 days worth
FILES=${directory}/Velocity2d_0{0480..1824}.h5
timeList=( )
count=480
for f in ${directory}/Velocity2d_0{0480..1824}.h5
do
    timeList+="$(($startTime + ($count * $dt))) "
    count=$[$count + 1]
done

#for f in $FILES
#do
#    ((i=i%ncore)); ((i++==0)) && wait    
#    # overwrite if the flag is set
#    # only process if the output doesn't exist (and overwrite false)
#    # we only check v, assume the equi u worked ok!
#    if [[ ${overwrite} = true || ! -f "${f}_v.nc" ]] ; then
#        echo ${f}        
#        process_file ${f} &
#    fi
#done

joined=$(join , ${timeList[@]})
string="time[time]={${joined%,}}"
echo $string

# we can now merge all the .nc files created
# we can * as we've made the number 0ddd in Thetis so they should list in order
#Velocity2d_03840.h5_u.nc
ncecat ${directory}/Velocity2d_0{0480..1824}.h5_u.nc -O "${directory}/Velocity2d_U.nc"
ncecat ${directory}/Velocity2d_0{0480..1824}.h5_v.nc -O "${directory}/Velocity2d_V.nc"

# then rename the record dimension as time
ncrename -d record,time ${directory}/Velocity2d_U.nc
ncrename -d record,time ${directory}/Velocity2d_V.nc

# adjust units of time to something sensible
ncap2 -s "${string}" ${directory}/Velocity2d_U.nc -O ${directory}/Velocity2d_U.nc
ncap2 -s "${string}" ${directory}/Velocity2d_V.nc -O ${directory}/Velocity2d_V.nc
# we then have to alter the time to be a float
ncap2 -s 'time=float(time)' --overwrite ${directory}/Velocity2d_U.nc ${directory}/Velocity2d_U.nc
ncap2 -s 'time=float(time)' --overwrite ${directory}/Velocity2d_V.nc ${directory}/Velocity2d_V.nc


