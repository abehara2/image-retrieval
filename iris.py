import numpy as np
import cv2
import os
import PIL
import argparse
import urllib

from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import VGG16 
from keras.models import Model


class ImageRetreival:

    class Image:
        def __init__(self, filename, map, mat):
            this.name = filename
            this.feature_map =  map
            this.mat = mat

    def __init__(self, search_string):
        self.myImages = []
        self.model = VGG16(weights='imagenet', include_top=False)
        self.search_string = search_string
        self.target_image = None

        os.chdir("./images")
        with os.scandir(path) as files:
             for file in files:
                #store these
                name = file.name
                mat = img_to_array(load_img(name, target_size=(224, 224)))

                img_data = np.expand_dims(image, axis=0)
                img_data = vgg16_preprocess(img_data)
                vgg16_feature = model_vgg16.predict(img_data)

                #store these
                feature_map = np.asarray(vgg16_feature).flatten()

                imObject = new Image(name, feature_map, mat)
                self.myImages.append(imObject)

    def url_to_image(self, url):
        resp = urllib.urlopen(url)
        image = np.asarray(bytearray(resp.read()),dtype='uint8')
        image = cv2.imdecode(image, cv2,IMREAD_COLOR)
        return image

    
    def get_search_results(self):
        urlKeyword = parse.quote(self.search_string)
        url = ‘https://www.google.com/search?hl=jp&q=' + urlKeyword + ‘&btnG=Google+Search&tbs=0&safe=off&tbm=isch’
        headers = {“User-Agent”: “Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0”,}
        request = req.Request(url=url, headers=headers)
        page = req.urlopen(request)
        html = page.read().decode(‘utf-8’)
        html = bs4.BeautifulSoup(html, “html.parser”)
        elems = html.select(‘.rg_meta.notranslate’)

        MAX_IMAGES = 5
        counter = 0

        urls = []

        for ele in elems:
            ele = ele.contents[0].replace(‘“‘,’’).split(‘,’)
            eledict = dict()
            num = e.find(‘:’)
            eledict[e[0:num]] = e[num+1:]
            imageURL = eledict[‘ou’]
            urls.append(imageURL)
            counter += 1

            if counter == 5:
                break
        for url in urls:
            im = url_to_image(url)
            cv2.imshow("Target Images", im)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()

        print("Which image # is closest to what you want?")
        inp_num = int(input())

        target_url = urls[inp_num - 1]
        image = url_to_image(url)
        image = cv2.resize(image, (224,224))

        img_data = np.expand_dims(image, axis=0)
        img_data = vgg16_preprocess(img_data)
        vgg16_feature = model_vgg16.predict(img_data)
        feature_map = np.asarray(vgg16_feature).flatten()

        self.target_image = new Image(target_url, feature_map, image)

    def get_nearest_neighbors(self):
        


    def main(self, query):
        ob = ImageRetreival(query)
        ob.get_search_results()
        ob.get_nearest_neighbors()


if __name == '__main__':

    parser.add_argument('--search', type = string, help = 'Google Images search query', default = "cow")
    args = parser.parse_args()
    main(arg.search)