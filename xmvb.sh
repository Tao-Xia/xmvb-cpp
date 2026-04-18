#!/bin/bash

#SBATCH -o %x-%j.slog
#SBATCH -e %x-%j.serr
#SBATCH -c 8

source /etc/xacs/config.sh
#source  /etc/profile
source /etc/profile.d/modules.sh

function getHostList()
{
    hostlist=$1

    prefix=$(echo $hostlist | awk -F '[][]' '{print $1}')
    ranges=$(echo $hostlist | awk -F '[][]' '{print $2}')
    
    #001,003-004
    echo $ranges | awk -F, '{for(i=1; i<=NF; i++) {print $i}}' | \
    while read range
    do
        echo $range | grep -q '-' 
        if [ $? -eq 0 ]; then
            begin=$(echo $range | awk -F '-' '{print $1}')
            end=$(echo $range | awk -F '-' '{print $2}')
            
            for n in $(seq -w $begin $end)
            do
                echo $prefix$n 
            done
        else
            echo $prefix$range
        fi
    done

}

# parse command line argument
ARGS=$(getopt -a -o v: --long version: -- "$@")

if [ $? != 0 ];then
        echo "argument parse error"
        exit 1
fi

eval set -- "${ARGS}"
while :
do
    case $1 in
        -v|--version)
            program_version=$2
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "unknown argument"
            exit 1
            ;;
    esac
shift
done

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch -c <threads> xmvb.sh [--version <version>] <input.xmi>" >&2
    exit 1
fi

INPUT_FILE="$1"

if [ -z "${program_version:-}" ]; then
    program_version="latest"
fi

PROGRAM_PATH="${XMVB_PROGRAM_PATH}/${program_version}"
PROGRAM_PATH_PC="/share/pc_apps/xmvb/${program_version}/"
PROGRAM_PATH_6526Y="/share/6526Y_apps/xmvb/${program_version}/"

module rm anaconda
module rm gcc/10.3.0

grep -q 'opt=gaussian' "${INPUT_FILE}" && GAUSS_OPT=1 || GAUSS_OPT=0

if [ ${GAUSS_OPT} -eq 0 ]; then
    if [ ${SLURM_JOB_PARTITION} == "pc" ]; then
        PROGRAM="${PROGRAM_PATH_PC}/bin/xmvb"
    elif [ ${SLURM_JOB_PARTITION} == "6526Y" ]; then
        module load gcc/13.2.0
        PROGRAM="${PROGRAM_PATH_6526Y}/bin/xmvb"
    else
        module load mpich/4.0.2-gcc13
#       PROGRAM="${PROGRAM_PATH}/bin/xmvb"
        PROGRAM="/export/home/fmying/softwares/6226r-package/xmvb/4.1/ilp64/bin/xmvb"
    fi
else
    PROGRAM="/share/xacs_apps/xmvb-gauss-gopt/xmvb-gauss.sh"
    if [ ${SLURM_JOB_PARTITION} == "pc" ]; then
        export PGI_FASTMATH_CPU=sandybridge
    else
        #module add gcc/13.2.0
        module load mpich/4.0.2-gcc13
    fi
    VBDIR="${PROGRAM_PATH}"
    export VBDIR
    
fi

JOB_ID=${SLURM_JOB_ID}
JOB_NAME=${SLURM_JOB_NAME}
NP=${SLURM_CPUS_PER_TASK}
NNODE=${SLURM_NNODES}

ulimit -c 0
ulimit -s unlimited
ulimit -v unlimited
export OMP_STACKSIZE=4G

SUBMIT_DIR=$(pwd)
RUNDIR=/job_dir/${JOB_NAME}_${JOB_ID}

mkdir -p ${RUNDIR}
chmod 700 ${RUNDIR}
# replace tab with space
sed -i $'s/\t/ /g' ${INPUT_FILE}
# cp the input file
cp ${INPUT_FILE} ${RUNDIR}/
# cp the initial guess
if [ -f ${INPUT_FILE%%.*}.gus ]; then
    cp ${INPUT_FILE%%.*}.gus ${RUNDIR}/
fi

if [ ${NNODE} -eq 1 ]; then   # single node
    export OMP_STACKSIZE=1G
    export OMP_NUM_THREADS=$NP
    # echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
    cd $RUNDIR
    #${PROGRAM} ${INPUT_FILE} 1>${INPUT_FILE%.*}.out 2>${INPUT_FILE%.*}.err
    #${PROGRAM} ${INPUT_FILE} | grep -v libuuid &> ${INPUT_FILE%.*}.cmdout
    new_version=0
    echo ${program_version} | egrep -q '^4' && new_version=1
    echo ${program_version} | egrep -q 'latest' && new_version=1
    if [ ${new_version} -eq 1 ]; then
        ${PROGRAM} -n ${NP} ${INPUT_FILE} 1> ${INPUT_FILE%%.*}.xmo  2> ${INPUT_FILE%.*}.cmdout
    else
        ${PROGRAM} ${INPUT_FILE} | grep -v libuuid &> ${INPUT_FILE%.*}.cmdout
    fi

else   # multiple nodes
    HOST_LIST_FILE=hostlist
    getHostList ${SLURM_JOB_NODELIST} > ${RUNDIR}/${HOST_LIST_FILE}
    tail -n+2 ${RUNDIR}/${HOST_LIST_FILE} | while read nodename
    do
        echo "making dir for $nodename"
        scp -r ${RUNDIR} ${nodename}:${RUNDIR}
        echo "made dir for $nodename"
    done

    cd $RUNDIR
    # within node: opemnp, between nodes: mpi
    mpirun -np $(( $NNODE )) \
      -env "OMP_STACKSIZE" "1G" \
      -env "OMP_NUM_THREADS" "${SLURM_CPUS_PER_TASK}" \
      -machinefile ${HOST_LIST_FILE} ${PROGRAM} ${INPUT_FILE} 1>${INPUT_FILE%.*}.out 2>${INPUT_FILE%.*}.err
fi

rm -f x*.int

# Fetch output file and other files possibly needed to starting directory
for f in $(find -maxdepth 1 -type f)
do
  if [ $(basename ${f}) != ${INPUT_FILE} ]; then
    cp ${f} $SUBMIT_DIR
  fi
done

cd ${SUBMIT_DIR}

touch $RUNDIR/_OK

