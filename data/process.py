import pyarrow.parquet as pq
import tarfile
from PIL import Image
import json

def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())

MAX_FILES_TO_READ = 136

FILES_TO_INCLUDE = 100000
files_included = 0

json_captions = { }

for i in range(MAX_FILES_TO_READ):

    filename = "{:05d}".format(i)

    try:
        table = pq.read_table(f'laion-high-resolution-output/{filename}.parquet', columns=['similarity', 'punsafe', 'LANGUAGE'])
        tar = tarfile.open("laion-high-resolution-output/" + filename + ".tar")
    except:
        continue

    to_include = []
    for index, row in table.to_pandas().iterrows():
        if float(row['similarity']) > 0.10 and float(row['punsafe']) < 0.15 and row['LANGUAGE'] == 'en':
            to_include.append(index)

    for j in to_include:
        try:
            data_element_filename = "{:09d}".format(j+10000*i)
            caption = tar.extractfile(data_element_filename + '.txt').read().decode('utf-8')
            if not isascii(caption):
                # double check language is English, if not, continue
                 continue
            image = Image.open(tar.extractfile(data_element_filename + '.jpg'))
            if not (image.width >= 1024 and image.height >= 1024):
                # check for corrupted images (all images should satisfy this condition)
                 continue
            new_element_filename = "{:09d}".format(files_included)
            image.save("/pine/scr/m/w/rwomick/laion-high-resolution/100k/" + new_element_filename +".jpg")
            json_captions[new_element_filename] = caption
            files_included += 1
            if files_included >= FILES_TO_INCLUDE:
                break
        except:
            pass

    if files_included >= FILES_TO_INCLUDE:
                break
    
with open("/pine/scr/m/w/rwomick/laion-high-resolution/100k/captions.json", "w") as capfile:
    json.dump(json_captions, capfile)

