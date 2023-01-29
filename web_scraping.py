import os
from nturl2path import url2pathname
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
import numpy as np
import pydub.exceptions
pydub.exceptions.decoder_exe = '/usr/local/bin/ffmpeg'


def get_html(url):
    response = requests.get(url)
    return response.text

chinese_url = "https://accent.gmu.edu/browse_language.php?function=find&language=chinese"

#
# get base url- accent page (chinese)
# go to content div
# For each p in content_div
#     find href and store in a list
#
# For each href in list:
#   call get_html
#   find mp3 tag
#   download with whatever package and add to folder
#
def scrape_mp3():
    with requests.Session() as req:
        soup = BeautifulSoup(get_html(chinese_url), 'html.parser')
        count = 1
        for contentDiv in soup.select('div', {'class': 'content'}):
            for pDiv in contentDiv.findAll('p'):
                for href in pDiv.findAll('a'):
                    if count == 35:
                        break
                    href = href.get("href")
                    link = "https://accent.gmu.edu/" + href
                    mp3_link = "https://accent.gmu.edu/soundtracks/cantonese" + str(count) + ".mp3"
                    download = req.get(mp3_link)
                    if download.status_code == 200:
                        with open("./data/chinese/" + str(count) + ".mp3", 'wb') as f:
                            f.write(download.content)
                    else:
                        print("FAILURE" + count)
                    count +=1

def mp3_to_numpy():
    # finds max length of all numpy representations of mp3s
    max_len = 0
    for file in os.listdir("./data/chinese/"):
        try:
            audio = AudioSegment.from_mp3("./data/chinese/" + file)
            audio_np = np.array(audio.get_array_of_samples())
            if len(audio_np) > max_len:
                max_len = len(audio_np)
        except:
            continue

    audio_list = np.zeros((len(os.listdir("./data/chinese/")), max_len))
    count = 0
    for file in os.listdir("./data/chinese/"):
        try:
            audio = AudioSegment.from_mp3("./data/chinese/" + file)
            audio_np = np.array(audio.get_array_of_samples())
        except:
            continue
        audio_list[count][:len(audio_np)] = audio_np
        count += 1

if __name__ == "__main__":
    #scrape_mp3()
    mp3_to_numpy()
