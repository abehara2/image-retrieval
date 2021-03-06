import numpy as np
import cv2
import os
import PIL
import argparse
import urllib
from sklearn.neighbors import NearestNeighbors
import bs4
from selenium import webdriver
import time, requests


from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input


from keras.applications.vgg16 import VGG16 
from keras.models import Model


class ImageRetreival:

    def __init__(self, search_string):
        self.myImages = []
        self.features = []
        self.model = VGG16(weights='imagenet', include_top=False)
        self.search_string = search_string
        self.target_image = None
        print('\n')
        print('##########################')
        print('Welcome to Image Retrieval')
        print('##########################')
        print('\n')
        os.chdir(".\images")
        with os.scandir(".") as files:
            for file in files:
                #store these
                name = file.name
                print(name)
                mat_pre = cv2.imread(name)
                mat = load_img(name, target_size=(224, 224))
                mat = img_to_array(mat)
                img_data = np.expand_dims(mat, axis=0)
                img_data = preprocess_input(img_data)
                vgg16_feature = self.model.predict(img_data)

                #store these
                feature_map = np.asarray(vgg16_feature).flatten()

                self.features.append(feature_map)
                imObject = self.Image(name, feature_map, mat_pre)
                self.myImages.append(imObject)
    
    def url_to_image(self, url):
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()),dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def get_search_results(self):
        print(os.getcwd)
        browser = webdriver.Chrome(executable_path='../chromedriver.exe')
        #browser = webdriver.Chrome()
        search_url = f"https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&q={self.search_string}"
        images_url = []

        # open browser and begin search
        browser.get(search_url)
        elements = browser.find_elements_by_class_name('rg_i')

        count = 0
        for e in elements:
            # get images source url
            e.click()
            time.sleep(1)
            element = browser.find_elements_by_class_name('v4dQwb')

            # Google image web site logic
            if count == 0:
                big_img = element[0].find_element_by_class_name('n3VNCb')
            else:
                big_img = element[1].find_element_by_class_name('n3VNCb')

            images_url.append(big_img.get_attribute("src"))
            
            count += 1

            # Stop get and save after 5
            if count == 5:
                break
        
        counter = 0
        for url in images_url:
            counter+=1
            reponse = requests.get(images_url[0])
            if reponse.status_code == 200:
                im = self.url_to_image(images_url[0])
                image = cv2.resize(im, (750,750))
                cv2.imshow("Sample Image " + str(counter), image)
                cv2.waitKey()
        
        #cv2.destroyAllWindows()
        
        print("\n")
        print("Which image # is closest to what you want?")
        inp_num = int(input())
        print("\n")

        target_url = images_url[inp_num - 1]
        image_pre = self.url_to_image(target_url)
        image = cv2.resize(image_pre, (224,224))

        img_data = np.expand_dims(image, axis=0)
        img_data = preprocess_input(img_data)
        vgg16_feature = self.model.predict(img_data)
        feature_map = np.asarray(vgg16_feature).flatten()

        self.target_image = self.Image(target_url, feature_map, image_pre)      

    def get_nearest_neighbors(self):
        num_neighbors = 3
        neighbors = NearestNeighbors(n_neighbors=num_neighbors).fit(self.features)
        distances, indices = neighbors.kneighbors([self.target_image.feature_map])
        for i in range(len(indices[0])):
            index = indices[0][i]
            cv2.imshow(self.myImages[index].name, self.myImages[index].mat)
            cv2.waitKey()
        cv2.destroyAllWindows()
        return

    class Image:
        def __init__(self, filename, fmap, mat):
            self.name = filename
            self.feature_map = fmap
            self.mat = mat


    def main(self):
        self.get_search_results()
        self.get_nearest_neighbors()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get a user-inputted search query')
    parser.add_argument('--search', type = str,  help = 'Google Images search query', default = "cow")
    args = parser.parse_args()
    ob = ImageRetreival(args.search)
    ob.main()