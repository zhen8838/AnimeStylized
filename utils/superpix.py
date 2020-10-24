from skimage import segmentation, color
from scipy.ndimage import find_objects
import numpy as np
import utils.feat_measure as measure


def slic(image, seg_num=200):
  seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1,
                                compactness=10, convert2lab=True,
                                start_label=0)
  image = color.label2rgb(seg_label, image, kind='avg', bg_label=-1)
  return image


def adaptive_slic(image, seg_num=200):
  seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1,
                                compactness=10, convert2lab=True,
                                start_label=0)
  image = adaptive_label2rgb(seg_label, image, kind='mix')
  return image


def adaptive_label2rgb(label_field, image, kind='mix', bg_label=-1, bg_color=(0, 0, 0)):
  out = np.zeros_like(image)
  labels = np.unique(label_field)
  bg = (labels == bg_label)
  if bg.any():
    labels = labels[labels != bg_label]
    mask = (label_field == bg_label).nonzero()
    out[mask] = bg_color
  for label in labels:
    mask = (label_field == label).nonzero()
    color: np.ndarray = None
    if kind == 'avg':
      color = image[mask].mean(axis=0)
    elif kind == 'median':
      color = np.median(image[mask], axis=0)
    elif kind == 'mix':
      std = np.std(image[mask])
      if std < 20:
        color = image[mask].mean(axis=0)
      elif 20 < std < 40:
        mean = image[mask].mean(axis=0)
        median = np.median(image[mask], axis=0)
        color = 0.5 * mean + 0.5 * median
      elif 40 < std:
        color = image[mask].median(axis=0)
    out[mask] = color
  return out


def switch_color_space(img, target):
  """
      RGB to target color space conversion.
      I: the intensity (grey scale), Lab, rgI: the rg channels of
      normalized RGB plus intensity, HSV, H: the Hue channel H from HSV
  """
  if target == 'HSV':
    return color.rgb2hsv(img)

  elif target == 'Lab':
    return color.rgb2lab(img)

  elif target == 'I':
    return color.rgb2grey(img)

  elif target == 'rgb':
    img = img / np.sum(img, axis=0)
    return img

  elif target == 'rgI':
    img = img / np.sum(img, axis=0)
    img[:, :, 2] = color.rgb2grey(img)
    return img

  elif target == 'H':
    return color.rgb2hsv(img)[:, :, 0]

  else:
    raise "{} is not suported.".format(target)


class HierarchicalGrouping(object):
  def __init__(self, img, img_seg, sim_strategy):
    self.img = img
    self.sim_strategy = sim_strategy
    self.img_seg = img_seg.copy()
    self.labels = np.unique(self.img_seg).tolist()

  def build_regions(self):
    self.regions = {}
    lbp_img = measure.generate_lbp_image(self.img)
    for label in self.labels:
      size = (self.img_seg == 1).sum()
      region_slice = find_objects(self.img_seg == label)[0]
      box = tuple([region_slice[i].start for i in (1, 0)] +
                  [region_slice[i].stop for i in (1, 0)])

      mask = self.img_seg == label
      color_hist = measure.calculate_color_hist(mask, self.img)
      texture_hist = measure.calculate_texture_hist(mask, lbp_img)

      self.regions[label] = {
          'size': size,
          'box': box,
          'color_hist': color_hist,
          'texture_hist': texture_hist
      }

  def build_region_pairs(self):
    self.s = {}
    for i in self.labels:
      neighbors = self._find_neighbors(i)
      for j in neighbors:
        if i < j:
          self.s[(i, j)] = measure.calculate_sim(self.regions[i],
                                                 self.regions[j],
                                                 self.img.size,
                                                 self.sim_strategy)

  def _find_neighbors(self, label):
    """
        Parameters
    ----------
        label : int
            label of the region
    Returns
    -------
        neighbors : list
            list of labels of neighbors
    """

    boundary = segmentation.find_boundaries(self.img_seg == label,
                                            mode='outer')
    neighbors = np.unique(self.img_seg[boundary]).tolist()

    return neighbors

  def get_highest_similarity(self):
    return sorted(self.s.items(), key=lambda i: i[1])[-1][0]

  def merge_region(self, i, j):

    # generate a unique label and put in the label list
    new_label = max(self.labels) + 1
    self.labels.append(new_label)

    # merge blobs and update blob set
    ri, rj = self.regions[i], self.regions[j]

    new_size = ri['size'] + rj['size']
    new_box = (min(ri['box'][0], rj['box'][0]),
               min(ri['box'][1], rj['box'][1]),
               max(ri['box'][2], rj['box'][2]),
               max(ri['box'][3], rj['box'][3]))
    value = {
        'box': new_box,
        'size': new_size,
        'color_hist':
            (ri['color_hist'] * ri['size']
             + rj['color_hist'] * rj['size']) / new_size,
        'texture_hist':
            (ri['texture_hist'] * ri['size']
             + rj['texture_hist'] * rj['size']) / new_size,
    }

    self.regions[new_label] = value

    # update segmentation mask
    self.img_seg[self.img_seg == i] = new_label
    self.img_seg[self.img_seg == j] = new_label

  def remove_similarities(self, i, j):

    # mark keys for region pairs to be removed
    key_to_delete = []
    for key in self.s.keys():
      if (i in key) or (j in key):
        key_to_delete.append(key)

    for key in key_to_delete:
      del self.s[key]

    # remove old labels in label list
    self.labels.remove(i)
    self.labels.remove(j)

  def calculate_similarity_for_new_region(self):
    i = max(self.labels)
    neighbors = self._find_neighbors(i)

    for j in neighbors:
      # i is larger than j, so use (j,i) instead
      self.s[(j, i)] = measure.calculate_sim(self.regions[i],
                                             self.regions[j],
                                             self.img.size,
                                             self.sim_strategy)

  def is_empty(self):
    return True if not self.s.keys() else False

  def num_regions(self):
    return len(self.s.keys())


def sscolor(image, seg_num=200, power=1,
            color_space='Lab', k=10, sim_strategy='CTSF'):

  img_seg = segmentation.felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
  img_cvtcolor = adaptive_label2rgb(img_seg, image, kind='mix')
  img_cvtcolor = switch_color_space(img_cvtcolor, color_space)
  S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
  S.build_regions()
  S.build_region_pairs()

  # Start hierarchical grouping

  while S.num_regions() > seg_num:

    i, j = S.get_highest_similarity()
    S.merge_region(i, j)
    S.remove_similarities(i, j)
    S.calculate_similarity_for_new_region()

  image = adaptive_label2rgb(S.img_seg, image, kind='mix')
  image = (image + 1) / 2
  image = image**power
  image = image / np.max(image)
  image = image * 2 - 1

  return image
