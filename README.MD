# ML coursera class

## Solutions for number of languages

+ matlab (octave)
+ python (tensor_flow)
+ F# (custom ML library)

## Run tensoFlow via docker


```
docker run -p 8888:8888 -it --rm b.gcr.io/tensorflow-udacity/assignments
//Windows: to edit code locally and run in docker container shared folder must be under User directory
docker run -p 8887:8888 -v /c/Users/v-maputi/dev/m-l:/usr/dev -it --rm b.gcr.io/tensorflow-udacity/assignments bash
```
