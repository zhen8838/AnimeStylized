from pathlib import Path
import shutil
import os
import webdataset as wds

if __name__ == "__main__":
  train = Path('/media/zqh/Documents/facelandmark_dataset/train')
  test = Path('/media/zqh/Documents/facelandmark_dataset/test')
  if not train.exists():
    train.mkdir()
  if not test.exists():
    test.mkdir()
  org1 = Path('/media/zqh/Documents/JOJO_face_crop_big')
  org2 = Path('/home/zqh/workspace/data512x512')

  test_ids = []
  train_ids = []

  for org in [org1, org2]:
    ids = list(set([p.stem for p in org.iterdir()]))
    n = len(ids)
    test_n = int(n * 0.1)
    for id in ids[:test_n]:
      test_ids.append(org / id)

    for id in ids[test_n:]:
      train_ids.append(org / id)

  for dst_root, ids in [(test, test_ids), (train, train_ids)]:
    total = len(ids)
    pattern = dst_root.as_posix() + f'-{str(total)}-%d.tar'
    with wds.ShardWriter(pattern, maxcount=5000, encoder=False) as f:
      for id in ids:
        with open(id.as_posix() + '.jpg', "rb") as stream:
          image = stream.read()
        with open(id.as_posix() + '.json', "rb") as stream:
          json = stream.read()
        key = id.name
        f.write({'__key__': key, 'jpg': image, 'json': json})

  """
  cd /media/zqh/Documents/facelandmark_dataset/
  sync
  tar --sort=name -cf train.tar train/
  sync
  tar --sort=name -cf test.tar test/
  sync
  rm -rf train test
  tar -tf train.tar | wc -l
  tar -tf test.tar | wc -l
  """
  # os.system("tar -tf /media/zqh/Documents/facelandmark_dataset/test.tar | wc -l")
