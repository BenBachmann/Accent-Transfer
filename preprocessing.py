import os
from nturl2path import url2pathname
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
import numpy as np
import pydub.exceptions
import pickle
# pydub.exceptions.decoder_exe = '/usr/local/bin/ffmpeg'


def get_html(url):
    response = requests.get(url)
    return response.text

chinese_url = "https://accent.gmu.edu/browse_language.php?function=find&language=chinese"
italian_url = "https://accent.gmu.edu/browse_language.php?function=find&language=italian"

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
    # downloads mp3s into a directory

    with requests.Session() as req:
        soup = BeautifulSoup(get_html(italian_url), 'html.parser')
        count = 1
        for contentDiv in soup.select('div', {'class': 'content'}):
            for pDiv in contentDiv.findAll('p'):
                for href in pDiv.findAll('a'):
                    if count == 35:
                        break
                    href = href.get("href")
                    link = "https://accent.gmu.edu/" + href
                    mp3_link = "https://accent.gmu.edu/soundtracks/italian" + str(count) + ".mp3"
                    download = req.get(mp3_link)
                    print(mp3_link)
                    if download.status_code == 200:
                        with open("./data/italian/" + str(count) + ".mp3", 'wb') as f:
                            f.write(download.content)
                    else:
                        print("FAILURE" + str(count))
                    count +=1

def mp3_to_numpy(language):
    # goes through all the mp3 files and returns a numpy array

    # finds max length of all numpy representations of mp3s
    max_len = 0
    for file in os.listdir("./data/{}/".format(language)):
        try:
            audio = AudioSegment.from_mp3("./data/{}/".format(language) + file)
            audio_np = np.array(audio.get_array_of_samples())
            if len(audio_np) > max_len:
                max_len = len(audio_np)
        except:
            continue

    # creates np array of all audio files
    audio_list = np.zeros((len(os.listdir("./data/{}/".format(language))), int(max_len / 2))) #divided by 10 here
    count = 0
    for file in os.listdir("./data/{}/".format(language)):
        try:
            audio = AudioSegment.from_mp3("./data/{}/".format(language) + file)
            audio_np = np.array(audio.get_array_of_samples())
        except:
            continue
        audio_np_short = audio_np[int(len(audio_np) / 2):2*int(len(audio_np) / 2)] #added this
        audio_list[count][:len(audio_np_short)] = audio_np_short #assigned to short here
        count += 1
    return audio_list

def pickle_array(arr):
    # Open a file for writing in binary mode
    with open("./chinese.pickle", "wb") as f:
        # Pickle the array and write it to the file
        pickle.dump(arr, f)

    # Open a file for reading in binary mode
    with open("./chinese.pickle", "rb") as f:
        # Load the pickled array from the file
        loaded_arr = pickle.load(f)

    print(loaded_arr)
    print(loaded_arr.shape)


if __name__ == "__main__":
    #pickle_array(scrape_mp3())
    pickle_array(mp3_to_numpy("chinese"))