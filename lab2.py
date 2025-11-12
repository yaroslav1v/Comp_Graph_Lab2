import numpy as np  
from PIL import Image, ImageOps 
from random import *  
from math import * 
 
def cord(x,y,x0,y0,x1,y1,x2,y2): 
    lambda0 = ((x-x2)*(y1-y2)-(x1-x2)*(y-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2)) 
    lambda1 = ((x0-x2)*(y-y2)-(x-x2)*(y0-y2))/((x0-x2)*(y1-y2) - (x1 - x2) * (y0-y2)) 
    lambda2 = 1.0 -lambda0 - lambda1 
    return(lambda0, lambda1, lambda2) 
def draw(x0,y0,z0,x1,y1,z1,x2,y2,z2, normal,buf): 
    light = [0,0,1] 
    cos = np.dot(light,normal) 
    if cos >=0: return 
    xmin= floor(min(x0,x1,x2)) 
    ymin = floor(min(y0,y1,y2)) 
    xmax = ceil(max(x0,x1,x2)) 
    ymax = ceil(max(y0,y1,y2)) 
     
    if (xmin <0): xmin = 0 
    if (ymin <0): xmin = 0 
    if (xmax >2000): xmin = 2000 
    if (ymax >2000): xmin = 2000  
    color=cos * -255 
    for x in range(int(xmin),int(xmax)+1): 
        for y in range(int(ymin),int(ymax)+1): 
            lambda0, lambda1, lambda2 =cord(x,y,x0,y0,x1,y1,x2,y2) 
            if lambda0>0 and lambda1>0 and lambda2>0: 
                z=lambda0*z0+lambda1*z1+lambda2*z2 
                if z< buf[y,x]: 
                    buf[y,x]=z 
                    img_mat[y,x]=color 
def calculate_normal(x0,y0,z0,x1,y1,z1,x2,y2,z2): 
    v1=[x1-x0,y1-y0,z1-z0] 
    v2=[x2-x0,y2-y0,z2-z0] 
    normalvector=np.cross(v1,v2) 
    norm = np.linalg.norm(normalvector) 
    vec = normalvector/ norm 
    return vec 
    
     
img_mat = np.zeros((2000, 2000,3), dtype = np.uint8)  
file = open('model_1.obj')  
 
vertices = [] 
poligons = [] 
buf=np.full((2000,2000),np.inf,dtype=np.float32) 
for  s in file: 
    s_list =  s.split() 
    if(s_list[0]=='v'): 
        vertices.append([float(s_list[1]), float(s_list[2]), float(s_list[3])]) 
    if(s_list[0]=='f'): 
        poligons.append([int(s_list[1].split('/')[0]), int(s_list[2].split('/')[0]), int(s_list[3].split('/')[0])]) 
  
for i in range (2000):  
    for j in range(2000):  
        img_mat[i,j,0] = 0  
 
for k in range(len(poligons)): 
    x0 = vertices[poligons[k][0]-1][0]  
    y0 = vertices[poligons[k][0]-1][1]  
    z0 = vertices[poligons[k][0]-1][2] 
    x1 = vertices[poligons[k][1]-1][0] 
    y1 = vertices[poligons[k][1]-1][1] 
    z1 = vertices[poligons[k][1]-1][2] 
    x2 = vertices[poligons[k][2]-1][0] 
    y2 = vertices[poligons[k][2]-1][1] 
    z2 = vertices[poligons[k][2]-1][2] 
    normal = calculate_normal(x0,y0,z0,x1,y1,z1,x2,y2,z2) 
    draw(x0* 10000 + 1000,y0 * 10000 + 500 ,z0,x1 * 10000 + 1000,y1 * 10000 + 500,z1,x2 * 10000 + 1000,y2 * 10000 + 500,z2,normal,buf) 
 
 
 
 
img = Image.fromarray(img_mat,mode='RGB') 
img= ImageOps.flip(img) 
img.save('img.png')