import random
import datetime
import os, shutil


def getDateTime():
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%m%d-%H%M")
        

def generateIndices(fraction=0.5, size=51967):
    currentDT = datetime.datetime.now()
    date = currentDT.strftime("%m%d-%H%M")
        
    num_data = int(fraction*size)
    return set(random.sample(range(0, size - 1), num_data))


def copySampledTrain():
    indices = generateIndices(0.5, 51967)
    for i in range(1,3):
        f = "train" + str(i) + ".txt"
        oldfile = fp(olddir, f)
        newfile = fp(newdir, f)
        print(oldfile, newfile)
        ct = 0
        with open(oldfile,"r",encoding="utf-8") as f1, open(newfile, "w",encoding="utf-8") as f2:
            for i, line in enumerate(f1):
                if i in indices:
                    f2.write(line)
                    
        

# creates full relative path
def fp(relPath, folder):
    return relPath + "/" + folder
        
def copyTestValSets():
    toMove = ["test0.txt", "test1.txt", "valid1.txt", "valid2.txt"]
    for f in toMove:
        shutil.copy(fp(olddir,f),newdir)



# get indices
olddir="data_fr"
indices = generateIndices()
samplesize = len(indices)
# get date time
dt = getDateTime()
# name new directory
newdir = "_".join([olddir, "randsamp" +  str(samplesize), str(dt)])
print(newdir + " created")
# make new directory
os.mkdir(newdir)
os.mkdir(newdir + "_output")
# move valid, test over to new folder
copyTestValSets()
copySampledTrain()
# create train1, train2
print("files copied.")

