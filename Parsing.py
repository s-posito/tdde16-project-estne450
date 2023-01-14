def parseDocuments(file):
    file = open(file)

    pre_line = ""
    abstracts = []
    abstract_temp = ""
    for line in file:
        if pre_line == ".W\n" and line != ".X\n":
            abstract_temp += line.replace("\n", " ")
            continue
        elif line == ".X\n":
            abstracts.append(abstract_temp)
            abstract_temp = ""
        pre_line = line
    file.close()
    return abstracts


def parseQueries(file):
    file = open(file)
    abstracts = []
    pre_line = ""
    abstract_temp = ""
    for line in file:
        if pre_line == ".W\n" and not line.startswith(".I "):
            abstract_temp += line.replace("\n", " ")
            continue
        elif line.startswith(".I ") and len(pre_line) > 0:
            abstracts.append(abstract_temp)
            abstract_temp = ""
        pre_line = line
    else:
        if len(abstract_temp) > 0:
            abstracts.append(abstract_temp)
    file.close()
    return abstracts


def parseResults(file):
    file = open(file)
    results = {}
    for line in file:
        arr = line.split()
        arr = [int(arr[0]), int(arr[1])]
        if arr[0] in results:
            results[arr[0]].append(arr[1])
        else:
            results[arr[0]] = [arr[1]]
    return results
