#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:32:13 2019

@author: zhouenna
"""
import cv2
from img_aug_funcs import * # NOQA


image_file="original_images/img2.jpg"


print("reading image...")
image=cv2.imread(image_file)


print("resizing...")
resize_image(image,450,400)






image_file="augmented_images/Resize-450*400.jpg"
print("current image:"+image_file)
image=cv2.imread(image_file)




print("cropping...")
crop_image(image,100,400,0,350)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,100,400,100,450)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,0,300,0,350)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,0,300,100,450)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,100,300,100,350)#(y1,y2,x1,x2)(bottom,top,left,right)


print("scaling...")

scale_image(image,0.3,0.3)
scale_image(image,0.7,0.7)
scale_image(image,2,2)
scale_image(image,3,3)


print("translating...")
translation_image(image,150,150)
translation_image(image,-150,150)
translation_image(image,150,-150)
translation_image(image,-150,-150)


print("rotating...")
rotate_image(image,90)
rotate_image(image,180)
rotate_image(image,270)


print("transforming...")
transformation_image(image)


print("padding...")
padding_image(image,100,0,0,0)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,0,100,0,0)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,0,0,100,0)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,0,0,0,100)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,100,100,100,100)#(y1,y2,x1,x2)(bottom,top,left,right)

print("flipping...")
flip_image(image,0)#horizontal
flip_image(image,1)#vertical
flip_image(image,-1)#both

print("superpixel...")
superpixel_image(image_file,100)
superpixel_image(image_file,50)
superpixel_image(image_file,25)
superpixel_image(image_file,75)
superpixel_image(image_file,200)



print("inverting...")
invert_image(image,255)
invert_image(image,200)
invert_image(image,150)
invert_image(image,100)
invert_image(image,50)




print("adding light...")
add_light(image,1.5)
add_light(image,2.0)
add_light(image,2.5)
add_light(image,3.0)
add_light(image,4.0)
add_light(image,5.0)
add_light(image,0.7)
add_light(image,0.4)
add_light(image,0.3)
add_light(image,0.1)


print("adding light with color...")
add_light_color(image,255,1.5)
add_light_color(image,200,2.0)
add_light_color(image,150,2.5)
add_light_color(image,100,3.0)
add_light_color(image,50,4.0)
add_light_color(image,255,0.7)
add_light_color(image,150,0.3)
add_light_color(image,100,0.1)


print("saturating...")
saturation_image(image,50)
saturation_image(image,100)
saturation_image(image,150)
saturation_image(image,200)

print("hueing")
hue_image(image,50)
hue_image(image,100)
hue_image(image,150)
hue_image(image,200)

print("multiplying..")
multiply_image(image,0.5,1,1)
multiply_image(image,1,0.5,1)
multiply_image(image,1,1,0.5)
multiply_image(image,0.5,0.5,0.5)

multiply_image(image,0.25,1,1)
multiply_image(image,1,0.25,1)
multiply_image(image,1,1,0.25)
multiply_image(image,0.25,0.25,0.25)

multiply_image(image,1.25,1,1)
multiply_image(image,1,1.25,1)
multiply_image(image,1,1,1.25)
multiply_image(image,1.25,1.25,1.25)

multiply_image(image,1.5,1,1)
multiply_image(image,1,1.5,1)
multiply_image(image,1,1,1.5)
multiply_image(image,1.5,1.5,1.5)


print("blurring..")
gausian_blur(image,0.25)
gausian_blur(image,0.50)
gausian_blur(image,1)
gausian_blur(image,2)
gausian_blur(image,4)

averageing_blur(image,5)
averageing_blur(image,4)
averageing_blur(image,6)

median_blur(image,3)
median_blur(image,5)
median_blur(image,7)

bileteralBlur(image,9,75,75)
bileteralBlur(image,12,100,100)
bileteralBlur(image,25,100,100)
bileteralBlur(image,40,75,75)


print("erosion..")
erosion_image(image,1)
erosion_image(image,3)
erosion_image(image,6)


print("dilation..")
dilation_image(image,1)
dilation_image(image,3)
dilation_image(image,5)



print("morphology")
opening_image(image,1)
opening_image(image,3)
opening_image(image,5)

closing_image(image,1)
closing_image(image,3)
closing_image(image,5)

morphological_gradient_image(image,5)
morphological_gradient_image(image,10)
morphological_gradient_image(image,15)

top_hat_image(image,200)
top_hat_image(image,300)
top_hat_image(image,500)

black_hat_image(image,200)
black_hat_image(image,300)
black_hat_image(image,500)


print("sharpen..")
sharpen_image(image)
print("emboss...")
emboss_image(image)

print("edge...")
edge_image(image,1)
edge_image(image,3)
edge_image(image,5)
edge_image(image,9)

print("addeptive gaussian noise...")
addeptive_gaussian_noise(image)

print("salt...")
salt_image(image,0.5,0.009)
salt_image(image,0.5,0.09)
salt_image(image,0.5,0.9)

print("paper...")
paper_image(image,0.5,0.009)
paper_image(image,0.5,0.09)
paper_image(image,0.5,0.9)

print("salt and paper...")
salt_and_paper_image(image,0.5,0.009)
salt_and_paper_image(image,0.5,0.09)
salt_and_paper_image(image,0.5,0.9)

print("constrast...") #this one spend alot of time
contrast_image(image,25)
contrast_image(image,50)
contrast_image(image,100)

print("edge detect canny...")
edge_detect_canny_image(image,100,200)
edge_detect_canny_image(image,200,400)

print("grey scale...")
grayscale_image(image)





















