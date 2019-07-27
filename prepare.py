from sklearn.model_selection import KFold
import glob
import numpy as np

fgs = np.array(glob.glob('/home/kunato/bg-removal/matting_human_dataset/matting/**/**/**.png'))

kf = KFold(n_splits=2, shuffle=True)

for i, data in enumerate(kf.split(fgs)):
    train_index, test_index = data
    print(train_index)
    y_train = fgs[train_index]
    y_test = fgs[test_index]
    with open(f'train_{i}.txt','w') as f:
        f.write('\n'.join(y_train))
    with open(f'test_{i}.txt','w') as f:
        f.write('\n'.join(y_test))