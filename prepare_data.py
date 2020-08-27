import json, os
import pandas as pd 
from shutil import copyfile
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

with open('data/all.json', 'r') as f:
    instance = json.load(f)

df = pd.DataFrame(instance['images'])
df_categories = pd.DataFrame(instance['categories']).loc[:, ['id', 'name']]
df_categories.columns = ['category_id', 'name']

categories = df.category_ids
X = df.id.values.reshape(-1, 1)
y = MultiLabelBinarizer().fit_transform(categories.values)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_index, test_index = list(sss.split(X, y))[0]

id_train = X[train_index, :].ravel()
id_test = X[test_index, :].ravel()

df_images = pd.DataFrame(instance['images'])
df_images_train = df_images.iloc[train_index, :]
df_images_test = df_images.iloc[test_index, :]

df_annotations = pd.DataFrame(instance['annotations'])
df_annotations_train = df_annotations[df_annotations.image_id.isin(id_train)]
df_annotations_test = df_annotations[df_annotations.image_id.isin(id_test)] 

if 11 < 2: # prepare the COCO data format
    instance_train = {'categories' : instance['categories']}
    instance_train['images'] = json.loads(df_images_train.to_json(orient='records'))
    instance_train['annotations'] = json.loads(df_annotations_train.to_json(orient='records'))

    with open('COCO/annotations/instances_train2014.json', 'w') as f:
        json.dump(instance_train, f, indent=4)

    instance_test = {'categories' : instance['categories']}
    instance_test['images'] = json.loads(df_images_test.to_json(orient='records'))
    instance_test['annotations'] = json.loads(df_annotations_test.to_json(orient='records'))

    with open('COCO/annotations/instances_val2014.json', 'w') as f:
        json.dump(instance_test, f, indent=4)

    for f in df_images_train.file_name.values:
        copyfile(
            os.path.join('data', f),
            os.path.join('COCO/train2014', f),
        )

    for f in df_images_test.file_name.values:
        copyfile(
            os.path.join('data', f),
            os.path.join('COCO/val2014', f),
        )

# For keras-frcnn
if 1 < 2:
    df_ = df_images_train.loc[:, ['id', 'file_name']]
    df_.columns = ['image_id', 'file_name']
    df = df_annotations_train.merge(
        df_, on='image_id'
    ).merge(
        df_categories, on='category_id'
    )

    lines = []
    for index, row in df.iterrows():
        lines.append(
            ','.join(
                [
                os.path.join('./train', row.file_name),
                str(row.bbox[0]),
                str(row.bbox[1]),
                str(row.bbox[0] + row.bbox[2]),
                str(row.bbox[1] + row.bbox[3]),
                row['name']]
            ) + "\n"
        )

    with open('train.txt', 'w') as f:
        f.writelines(lines)

    df_ = df_images_test.loc[:, ['id', 'file_name']]
    df_.columns = ['image_id', 'file_name']
    df = df_annotations_test.merge(
        df_, on='image_id'
    ).merge(
        df_categories, on='category_id'
    )

    lines = []
    for index, row in df.iterrows():
        lines.append(
            ','.join(
                [
                os.path.join('./test', row.file_name),
                str(row.bbox[0]),
                str(row.bbox[1]),
                str(row.bbox[0] + row.bbox[2]),
                str(row.bbox[1] + row.bbox[3]),
                row['name']]
            ) + "\n"
        )

    with open('test.txt', 'w') as f:
        f.writelines(lines)