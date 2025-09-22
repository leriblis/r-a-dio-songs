# r-a-dio-songs

Parse and save names of the last played songs on http://r-a-d.io.

I have also added some basic analysis of the stuff that was played over the years in ```stats/R-a-dio overview.ipynb```

## Installation

On Ubuntu run:
```
sudo apt update && sudo apt install python3 python3-venv xz-utils git
./autoupdate.sh
```

`autoupdate.sh` creates a virtual environment, bootstraps `pip` with
`python -m ensurepip` when required, installs the dependencies listed in
`requirements.txt`, and then refreshes the local song database.
