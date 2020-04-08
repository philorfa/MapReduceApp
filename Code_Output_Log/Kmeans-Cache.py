import findspark
findspark.init()
import pandas as pd
import pyspark

from pyspark import SparkContext, SparkConf

from math import radians, cos, sin, asin, sqrt

conf = SparkConf().setAppName('kMeans')
sc = SparkContext(conf=conf)

tripdata=sc.textFile("hdfs://master:9000/yellow_tripdata_1m.csv").map(lambda line: (float(line.split(',')[3]), float(line.split(',')[4]))).filter(lambda line: (-75.0<=line[0] and line[0]<=-73.0) and (40.0<=line[1] and line[1]<=41.0))
tripdata.cache()

centroids=tripdata.take(5)

def harv(lat_longt,centr0,centr1):
    dlon=lat_longt[0]-centr0
    dlat=lat_longt[1]-centr1
    a = sin(dlat/2)**2 + cos(lat_longt[1]) * cos(centr1) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 
    return c * r

def closerto(latitude_longtitude,centers):
        closer = 0
        minimum= float("+inf")
        harvesine = 0
        for i in range(len(centers)):
                harvesine =harv(latitude_longtitude,centers[i][0],centers[i][1])
                if(minimum>harvesine):
                        minimum = harvesine
                        closer = i

        return closer

for i in range(3):
    closer = tripdata.map(lambda p: (closerto(p,centroids),(p,1)))
    tmpcloser=closer.reduceByKey(lambda first,sec: ((first[0][0]+sec[0][0],first[0][1]+sec[0][1]),first[1]+sec[1]))
    if(i==2):
        centroids1=tmpcloser.map(lambda calc: (calc[0]+1,(calc[1][0][0]/calc[1][1],calc[1][0][1]/calc[1][1])))
        centroids=centroids1.take(5)
    else:
        centroids1=tmpcloser.mapValues(lambda calc: (calc[0][0]/calc[1],calc[0][1]/calc[1]))
        centroids=centroids1.collectAsMap()

for i in range(5):
        print(centroids[i][0] , centroids[i][1])

centroids1.coalesce(1).saveAsTextFile("hdfs://master:9000/output")
