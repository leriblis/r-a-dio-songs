import requests
import time

# URL of the MP3 stream
stream_url = "https://relay0.r-a-d.io/main.mp3"

# Path to where you want to save the MP3 file
file_path = "saved_stream.mp3"

time_limit = 60 * 60  # 1 hour

response = requests.get(stream_url, stream=True)
start_time = time.time()

if response.status_code == 200:
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                if time.time() - start_time > time_limit:
                    print("Reached the time limit for recording.")
                    break
else:
    print(f"Error retrieving the stream. Status code: {response.status_code}")

print(f"MP3 stream has been saved to {file_path}")

# Make sure to close the connection
response.close()
