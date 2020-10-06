#----This script creates/appends a more readable version of the file "detections.csv"----#

#!/usr/bin/env python
# coding: utf-8

# import packages
import os
import pandas as pd
import csv
from datetime import datetime

# rks code - set up mongodb and define row insert function
import pymongo
# myclient = pymongo.MongoClient("mongodb://localhost:27017/")
myclient = pymongo.MongoClient("mongodb://api1.csgtechnical.com, api2.csgtechnical.com, api3.csgtechnical.com", replicaSet='rs0' )
#
detectionsdb = myclient["detectionsdb"]
mycol = detectionsdb["counts"]
try:
  myindex1 = mycol.create_index([("camera_name", 1)])
except:
  print("Cannot create index within mongodb collection 'counts' ")
def insert_mongodb(row):
  mongo_record = {"timestamp":row[0], "image":row[1], "car":int(row[2]), "person":int(row[3]), "bicycle":int(row[4]), "motorcycle":int(row[5]), "bus":int(row[6]), "truck":int(row[7])}
  imagetime = row[1][0:13]
  cameraname = row[1][14:14+(row[1][14:99].find("_"))] # from char 14 to the position of next underscore
  mongo_record['image_timestamp'] = (datetime.fromtimestamp(int(imagetime)/1000))
  mongo_record['camera_name'] = cameraname
  z = mycol.insert_one(mongo_record)

# read csv file with image detections from pretrained model
df = pd.read_csv('./output_folder/detections.csv')

# group by image and type of object and perform counts
counts = df.groupby(['image_id', 'object']).count().score

# filter rows with the objects of interest
counts = counts[counts.index.get_level_values(1).isin(['bicycle', 'car', 'person', 'motorcycle', 'bus', 'truck'])]

# create or append a csv file with counts per object of interest
timestamp = datetime.now() #Return the current local date and time
if os.path.isfile('./output_folder/report.csv'):
  with open('./output_folder/report.csv', 'a') as file:
    writer = csv.writer(file, delimiter=",", lineterminator="\n")
   
    car = 0
    person = 0
    bicycle = 0
    motorcycle = 0
    bus = 0
    truck = 0
   
    image = counts.index[0][0]
    for object in range(len(counts)):
      if counts.index[object][0] == image:
        if counts.index[object][1] == 'car':
          car = counts[object]
          continue
        if counts.index[object][1] == 'person':
          person = counts[object]
          continue
        if counts.index[object][1] == 'bicycle':
          bicycle = counts[object]
          continue  
        if counts.index[object][1] == 'motorcycle':
          motorcycle = counts[object]
          continue  
        if counts.index[object][1] == 'bus':
          bus = counts[object]
          continue  
        if counts.index[object][1] == 'truck':
          truck = counts[object]
      else:
#        print(timestamp, image, car, person, bicycle, motorcycle, bus, truck, sep=",")
        row = [timestamp, image, car, person, bicycle, motorcycle, bus, truck]
        insert_mongodb(row) # rks code
        writer.writerow(row)
        car = 0
        person = 0
        bicycle = 0
        motorcycle = 0
        bus = 0
        truck = 0
       
        image = counts.index[object][0]    
        if counts.index[object][1] == 'car':
          car = counts[object]
          continue
        if counts.index[object][1] == 'person':
          person = counts[object]
          continue
        if counts.index[object][1] == 'bicycle':
          bicycle = counts[object]
          continue  
        if counts.index[object][1] == 'motorcycle':
          motorcycle = counts[object]
          continue  
        if counts.index[object][1] == 'bus':
          bus = counts[object]
          continue  
        if counts.index[object][1] == 'truck':
          truck = counts[object]
         
#    print(timestamp, image, car, person, bicycle, motorcycle, bus, truck, sep=",")
    row = [timestamp, image, car, person, bicycle, motorcycle, bus, truck]
    insert_mongodb(row) # rks code
    writer.writerow(row)
else:
  with open('./output_folder/report.csv', 'a') as file:
    file.write('timestamp,image,car,person,bicycle,motorcycle,bus,truck\n')
    writer = csv.writer(file, delimiter=",", lineterminator="\n")
   
    car = 0
    person = 0
    bicycle = 0
    motorcycle = 0
    bus = 0
    truck = 0
   
    image = counts.index[0][0]
    for object in range(len(counts)):
      if counts.index[object][0] == image:
        if counts.index[object][1] == 'car':
          car = counts[object]
          continue
        if counts.index[object][1] == 'person':
          person = counts[object]
          continue
        if counts.index[object][1] == 'bicycle':
          bicycle = counts[object]
          continue  
        if counts.index[object][1] == 'motorcycle':
          motorcycle = counts[object]
          continue  
        if counts.index[object][1] == 'bus':
          bus = counts[object]
          continue  
        if counts.index[object][1] == 'truck':
          truck = counts[object]
      else:
#        print(timestamp, image, car, person, bicycle, motorcycle, bus, truck, sep=",")
        row = [timestamp, image, car, person, bicycle, motorcycle, bus, truck]
        insert_mongodb(row) # rks code
        writer.writerow(row)
        car = 0
        person = 0
        bicycle = 0
        motorcycle = 0
        bus = 0
        truck = 0
       
        image = counts.index[object][0]    
        if counts.index[object][1] == 'car':
          car = counts[object]
          continue
        if counts.index[object][1] == 'person':
          person = counts[object]
          continue
        if counts.index[object][1] == 'bicycle':
          bicycle = counts[object]
          continue  
        if counts.index[object][1] == 'motorcycle':
          motorcycle = counts[object]
          continue  
        if counts.index[object][1] == 'bus':
          bus = counts[object]
          continue  
        if counts.index[object][1] == 'truck':
          truck = counts[object]
         
#    print(timestamp, image, car, person, bicycle, motorcycle, bus, truck, sep=",")
    row = [timestamp, image, car, person, bicycle, motorcycle, bus, truck]
    insert_mongodb(row) # rks code
    writer.writerow(row)
   
file.close()
