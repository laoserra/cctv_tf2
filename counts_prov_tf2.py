import pandas as pd
import os
from datetime import datetime

# ======= Count objects of interest =========

# Dataframe with image detections from pretrained model
df = pd.read_csv("./detections_2020-09-22T11 55 41.csv")
df = df.iloc[:,1:]

# group by image and type of object and perform counts
objects_of_interest = ['bicycle', 'car', 'person', 'motorcycle', 'bus', 'truck']
df = df[df.object.isin(objects_of_interest)]
df = df[['image_id', 'object']]
df1 = df.groupby(['image_id', 'object']).size().to_frame('counts').reset_index()

# transpose table with objects as columns
df1 = df1.pivot_table(index='image_id', columns='object', values='counts', fill_value=0)

# add absent columns with zero value
absent_objects = [obj for obj in objects_of_interest if obj not in df1.columns]

if absent_objects:
    for obj in absent_objects:
        df1[obj] = 0

df1['timestamp'] = datetime.now() #Return the current local date and time
df1 = df1.reset_index()
# reorder columns
df1 = df1[['timestamp','image_id','car','person','bicycle','motorcycle','bus','truck']]
df1.columns.name = None

# append dataframe to csv report
if os.path.isfile('./report.csv'):
    df1.to_csv('./report.csv', index=False, mode='a', header=False)
else:
    df1.rename(columns={'image_id':'image'}, inplace=True)
    df1.to_csv('./report.csv', index=False)
