from OpenGL.GL import *
import numpy
from PIL import Image
import pygame

class LoadTextures:
    def __init__(self, json_data):

        textures_count = len(json_data["textures"])

        print("loading textures")
        print("textures count ", textures_count)

        self.textures = []
        self.textures_ids_map = {}

        glEnable(GL_TEXTURE_2D)
        self.textures = glGenTextures(textures_count)

        for i in range(textures_count):
            texture_file_name = str(json_data["textures"][i]["file_name"])
            texture_id        = int(json_data["textures"][i]["id"])

            print("loading texture ", texture_id, texture_file_name)
 
            self.textures_ids_map[texture_id] = i

            texture_raw  = pygame.image.load(texture_file_name)
            texture_data = pygame.image.tostring(texture_raw, "RGB", 1)

            height = texture_raw.get_height()
            width  = texture_raw.get_width()

            '''
            im = Image.open(texture_file_name).transpose( Image.FLIP_TOP_BOTTOM )

            #raw_data = im.tostring()
            #texture_raw = numpy.fromstring(raw_data, numpy.uint8)

            

            texture_raw =texture_raw.reshape(height*width)
            '''

            
          
            

            glBindTexture(GL_TEXTURE_2D, self.textures[i])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)




    def get_textures_count(self):
        return len(self.textures)

    def get(self, id):
        idx = self.textures_ids_map[id]
        return self.textures[idx]
  
    def get_idx(self, idx):
        return self.textures[idx]
