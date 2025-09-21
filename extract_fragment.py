import sys
from pydub import AudioSegment
from argparse import ArgumentParser


def parse_time(time_str):
    minutes, seconds = map(int, time_str.split(":"))
    return (minutes * 60 + seconds) * 1000  # convert time to milliseconds


def extract_fragment(start_time, end_time, input_file, output_file):
    audio = AudioSegment.from_mp3(input_file)
    start_time_ms = parse_time(start_time)
    end_time_ms = parse_time(end_time)
    extract = audio[start_time_ms:end_time_ms]
    extract.export(output_file, format="mp3")


def main():
    parser = ArgumentParser(description="Extract a fragment from an MP3 file")
    parser.add_argument(
        "-s",
        "--start",
        required=True,
        help="Start time of the fragment (format: MM:SS)",
    )
    parser.add_argument(
        "-e", "--end", required=True, help="End time of the fragment (format: MM:SS)"
    )
    parser.add_argument("input_mp3", help="Input MP3 file")
    args = parser.parse_args()

    extract_fragment(args.start, args.end, args.input_mp3, "extracted.mp3")


if __name__ == "__main__":
    main()
