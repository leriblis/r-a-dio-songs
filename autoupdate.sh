#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
PYSCRIPT=${BASEDIR}/parse_radio.py
LOGNAME=${BASEDIR}/$(date +"%d_%m_%y_parse.log")

cd $BASEDIR
xz -d songs_db.json.xz
python $PYSCRIPT update &> $LOGNAME
xz -z songs_db.json
git add . --all
git commit -m "$(date +"%d-%m-%y update")"
git push -u origin master
echo "$(date +"%d-%m-%y") r-a-d.io update completed"
