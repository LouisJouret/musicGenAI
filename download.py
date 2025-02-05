import requests
from bs4 import BeautifulSoup
import os

# Base URL for the MIDI files directory
BASE_URL = "https://www.midiworld.com/files/"

# Function to fetch and download MIDI files from a given directory


def download_midi_files(directory_number):
    directory_url = f"{BASE_URL}{directory_number}/"

    response = requests.get(directory_url)
    if response.status_code != 200:
        print(f"Failed to fetch: {directory_url}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    download_links = soup.find_all('a', href=True, text="download")

    # Create a folder for MIDI files
    os.makedirs("MIDI_Files", exist_ok=True)

    for link in download_links:
        # This is the direct MIDI file download link
        midi_download_url = link['href']
        # Naming the file
        midi_filename = f"{midi_download_url.split('/')[-1]}.mid"

        # Download the MIDI file
        midi_response = requests.get(midi_download_url)
        if midi_response.status_code == 200:
            midi_path = os.path.join("midi_files", midi_filename)
            with open(midi_path, 'wb') as f:
                f.write(midi_response.content)
            print(f"Downloaded: {midi_filename}")
        else:
            print(f"Failed to download: {midi_filename}")


if __name__ == "__main__":
    for i in range(1, 300):
        download_midi_files(str(i))
