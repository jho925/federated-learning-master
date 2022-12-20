def shuffle_split(infilename, outfilename1, outfilename2):
    from random import shuffle

    with open(infilename, 'r') as f:
        lines = f.readlines()

    shuffle(lines)

    lines = [l.split(',')[0] + '\n' for l in lines]

    with open(outfilename1, 'w') as f:
        f.writelines(lines[:10089])
    with open(outfilename2, 'w') as f:
        f.writelines(lines[10089:])


shuffle_split('data/total_labels.csv', 'data/total_train.csv','data/total_test.csv')