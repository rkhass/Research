def get_indices(list_of_id):
    current_ID = ''
    current_subset = []
    indices = []

    for index, ID_name in enumerate(list_of_id):
        if ID_name == current_ID:
            current_subset.append(index)
        else:
            if index > 0: 
                indices.append(current_subset) # finish the subset and start new one
            current_ID = ID_name
            current_subset = [index]

    indices.append(current_subset) # for the last one

    return indices   

def sort_indices(arr):
    arr = [[i, int(i[3:])] for i in arr]
    return [i[0] for i in sorted(arr, key=lambda dt: dt[1])]

def construct_sentences(data, column_name, dropna=True):
    sentences = []

    series = data[column_name]
    uniques = sort_indices(data.ID.unique())
    indices = get_indices(data.ID)

    if not dropna:
        series = series.fillna('NAN' + column_name)

    for index, ID_name in tqdm(enumerate(uniques), total=len(uniques)):
        if dropna:
            sentences.append(list(series.iloc[indices[index]].dropna()))
        else:
            sentences.append(list(series.iloc[indices[index]]))

    sentences = [' '.join(sent) for sent in sentences]

    return sentences

def adaptive_boards(arr, N=2000):
    cur_boarder = 0
    boards = [0.]
    counter = 0
    counters = []
    for cur_val in arr:
        if counter < N:
            cur_boarder = cur_val
            counter += 1
        else:
            if cur_val == cur_boarder:
                counter += 1
            else:
                counters.append(counter)
                boards.append(cur_boarder)
                cur_boarder = cur_val
                counter = 1 

    counters.append(counter)
    boards.append(cur_boarder)
    return boards, counters

def save_features(train, test, name):
    train.to_pickle('../Features/Train/' + name + '.pkl')
    test.to_pickle('../Features/Test/' + name + '.pkl')