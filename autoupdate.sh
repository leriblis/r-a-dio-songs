#!/usr/bin/env bash

set -e  # Exit script if any command fails

BASEDIR=$(dirname "$0")
VENV_DIR=${BASEDIR}/.venv
PYSCRIPT=${BASEDIR}/parse_radio.py
LOGNAME=${BASEDIR}/$(date +"%Y_%m_%d_parse.log") # Updated with hours, minutes, and seconds to prevent overwriting logs from the same day

cd "$BASEDIR"

# Create virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Error: python3 is not installed or not on PATH." >&2
        exit 1
    fi

    if ! python3 -m venv "$VENV_DIR"; then
        echo "Error: failed to create virtual environment in $VENV_DIR." >&2
        echo "On Debian/Ubuntu, install the python3-venv package: sudo apt install python3-venv" >&2
        exit 1
    fi
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Error: $VENV_DIR/bin/python not found." >&2
    exit 1
fi

# Ensure pip is available inside the virtual environment
if ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "Pip not found in the virtual environment; bootstrapping with ensurepip..."
    if ! "$VENV_DIR/bin/python" -m ensurepip --upgrade; then
        echo "Error: failed to install pip via ensurepip." >&2
        exit 1
    fi
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

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
"$VENV_DIR/bin/python" "$PYSCRIPT" update &> "$LOGNAME"
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
