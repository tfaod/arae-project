#!/bin/bash
cd $(dirname "$0")

#echo "$(dirname "$0")"

SOURCEDIR=$1
#"/home/team/cs224nfinalproject/yelp/$1"
SAMPLESIZE=$2


#echo "/home/team/cs224nfinalproject/yelp/$SOURCEDIR"
if [[ -d /home/team/cs224nfinalproject/yelp/${SOURCEDIR} ]]; then
    DESTDIR=/home/team/cs224nfinalproject/yelp/${SOURCEDIR}_sample_${SAMPLESIZE}
    echo "Creating directories for ${DESTDIR}"
    mkdir -p ${DESTDIR}
    mkdir -p ${DESTDIR}_output
    
    for FILE in $(ls  ${SOURCEDIR}) ; do
	echo "Creating file for ${FILE}"
	DEST=${DESTDIR}/${FILE}
	touch ${DEST}
	head -n $(( SAMPLESIZE + 0)) ${SOURCEDIR}/${FILE} >${DEST}
    done
else
    echo "This is not a directory. Try again."
fi

echo "Creating new file ${SOURCEDIR}_sample.txt with ${SAMPLESIZE} sampled"
