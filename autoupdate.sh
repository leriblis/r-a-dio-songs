#!/usr/bin/env bash

set -e  # Exit script if any command fails

BASEDIR=$(dirname "$0")
PYSCRIPT=${BASEDIR}/parse_radio.py
LOGNAME=${BASEDIR}/$(date +"%Y_%m_%d_parse.log") # Updated with hours, minutes, and seconds to prevent overwriting logs from the same day

cd "$BASEDIR"
# Check if songs_db.json.xz exists
if [ -f songs_db.json.xz ]; then
    # Decompress it and overwrite songs_db.json if it already exists
    xz -d -f songs_db.json.xz
elif [ -f songs_db.json ]; then
    # Warn that we're using an existing songs_db.json
    echo "Warning: songs_db.json.xz does not exist. Using existing songs_db.json."
else
    # If neither file exists, exit with an error
    echo "Error: neither songs_db.json.xz nor songs_db.json exists."
    exit 1
fi

echo "Starting to update..."

# Run the Python script and capture its exit code
python "$PYSCRIPT" update &> "$LOGNAME"
echo "Python script finished. Check the log at $LOGNAME for details."
EXIT_CODE=$?

# Check if the Python script failed
if [ $EXIT_CODE -ne 0 ]; then
  echo "Python script failed with exit code $EXIT_CODE. Check the log at $LOGNAME for details."
  exit $EXIT_CODE
fi

xz -z songs_db.json
git add . --all
git commit -m "$(date +"%d-%m-%y update")"
git push -u origin master
echo "$(date +"%d-%m-%y") r-a-d.io update completed"
