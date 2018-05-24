import numpy as np
import sklearn

def readFile(path):
    file = open(path,'r',encoding="utf8")
    lines = []
    for i,line in enumerate(file):
        if i==0:
            continue
        lines.append(line.strip('\n'))
    return lines
        

def makeMovies():
    file = readFile('./dataset/movies.csv')
    x = np.zeros((len(file),3),dtype=np.chararray)
    for i,line in enumerate(file):
        tokens = line.strip('\n').split(',')
        x[i][0] = tokens[0]
        if len(tokens)>3:
            x[i][1] = tokens[1]+tokens[2]
            x[i-1][2] = tokens[3]
        else:
            x[i][1] = tokens[1]
            x[i][2] = tokens[2]
    return x

def makeRatings():
    file = readFile('./dataset/ratings.csv')
    x = np.zeros((len(file),4),dtype=np.chararray)
    for i,line in enumerate(file):
        tokens = line.split(',')
        x[i][0] = tokens[0]
        x[i][1] = tokens[1]
        x[i][2] = tokens[2]        
        x[i][3] = tokens[3]
    return x

def makeLinks():
    file = readFile('./dataset/links.csv')
    x = np.zeros((len(file),3),dtype=np.chararray)
    for i,line in enumerate(file):
        tokens = line.split(',')
        x[i][0] = tokens[0]
        x[i][1] = tokens[1]
        x[i][2] = tokens[2]
    return x

def makeTags():
    file = readFile('./dataset/links.csv')
    x = np.zeros((len(file),4),dtype=np.chararray)
    for i,line in enumerate(file):
        tokens = line.split(',')
        x[i][0] = tokens[0]
        x[i][1] = tokens[1]
        x[i][2] = tokens[2]       
        x[i][3] = tokens[3]
    return x

print(makeMovies())
print(makeRatings())
print(makeLinks())
print(makeTags())