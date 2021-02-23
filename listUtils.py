def saveList(listToSave, fname):
    with open(fname, 'w') as f:
         for item in listToSave:
           f.write("%s\n" % item)

def loadList(fname):
    listToLoad = [];
    with open(fname, 'rt') as f:
         for line in f:
            listToLoad.append(line.rstrip('\n'));
    return listToLoad;
