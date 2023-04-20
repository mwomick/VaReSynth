import pyarrow.parquet as pq
import tarfile
from PIL import Image
import skimage.measure    
import numpy as np

def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())

MAX_FILES_TO_READ = 136

FILES_TO_INCLUDE = 10
files_included = 0

valid_words = open("okay-words.txt", 'rt').readlines()
bad_words = set(open("bad-words.txt", 'rt').readlines())

out_captions = []
included_ids = []

for i in range(MAX_FILES_TO_READ):
    print("======================================================================")
    print("Reading file " + "{:05d}".format(i) + ".parquet")
    print("======================================================================")

    filename = "{:05d}".format(i)

    try:
        table = pq.read_table(f'laion-high-resolution-output/{filename}.parquet', columns=['similarity', 'punsafe', 'LANGUAGE'])
        tar = tarfile.open("laion-high-resolution-output/" + filename + ".tar")
    except:
        continue

    to_include = []
    for index, row in table.to_pandas().iterrows():
        if float(row['similarity']) > 0.15 and float(row['punsafe']) < 0.05 and row['LANGUAGE'] == 'en':
            to_include.append(index)

    print("Found " + str(len(to_include)) + " candidate image-caption pairs.")

    for j in to_include:
        try:
            data_element_filename = "{:09d}".format(j+10000*i)
            caption = tar.extractfile(data_element_filename + '.txt').read().decode('utf-8')
            if not isascii(caption):
                # double check language is English, if not, continue
                 continue
            
            split = caption.split(" ")
            if not bad_words.isdisjoint(set(split)):
                 print("\tNSFW caption - \"" + caption + "\"")
                 continue
            
            valid_count = 0
            for word in split:
                if valid_words.count(word) > 0:
                    valid_count += 1
            if valid_count / len(split) < .60:
                 print("\tInvalid caption - \"" + caption + "\"")
                 continue

            image = Image.open(tar.extractfile(data_element_filename + '.jpg'))
            if not (image.width >= 1024 and image.height >= 1024):
                # check for corrupted images (all images should satisfy this condition)
                 continue
            
            entropy = skimage.measure.shannon_entropy(np.array(image))
            if entropy < 5:
                print("\tLow entropy image detected (Shannon entropy: " + str(entropy) + ")")
                continue

            new_element_filename = "{:09d}".format(files_included)
            image.save("/pine/scr/m/w/rwomick/laion-high-resolution/10_test/" + new_element_filename +".jpg")
            out_captions.append((new_element_filename, caption))
            included_ids.append(j+10000*i)
            files_included += 1
            if files_included >= FILES_TO_INCLUDE:
                break
        except:
            pass

    if files_included >= FILES_TO_INCLUDE:
                break
    
with open("/pine/scr/m/w/rwomick/laion-high-resolution/10_test/captions.csv", "a+") as capfile:
    capfile.write("Item, Caption")
    for caption in out_captions:
        capfile.write(caption[0] + ", " + caption[1])

with open("/pine/scr/m/w/rwomick/laion-high-resolution/10_test/included_ids.txt", "a+") as idfile:
    for id in included_ids:
        idfile.write("{:09d}".format(id))

# TODO: Make more repeatable (e.g. create a file specifying the original URLs of the images)
# TODO: Test how 0.05 filtering works for filtering more rigorously against unsafe content (compare with original 0.15)