import os
import numpy as np
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from modules.clip import clip
from torchvision import transforms

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('.', ' ').replace(';', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ').replace('?', '').replace('\'s', ' \'s').replace('s\'', ' ').replace('n\'t', ' not').replace('"', ' ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome datasets
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class GasDataset(Dataset):
    def __init__(self, split, dataroot, transform=None, nans=2, max_length=48):
        super(GasDataset).__init__()
        assert split in ['train', 'test', 'val']
        self.root_dir = dataroot
        self.split = split
        self.transform = transform
        self.class_to_idx = {'y': 1, 'n': 0}
        self.num_ans_candidates = nans
        self.max_length = max_length

        self.entries = []
        entry = {}
        caption_path = os.path.join(dataroot, 'annotations', "%s_captions.json" % self.split)
        captions = json.load(open(caption_path))['captions']
        for cap in captions:
            img_id = cap['image_id']
            caption = cap['caption']
            label_str = cap['label']
            label_id = self.class_to_idx[label_str]
            img_path = '/'.join([self.split, label_str, img_id+".jpeg"])
            entry['img_path'] = img_path
            entry['label_id'] = label_id
            entry['caption'] = caption
            self.entries.append(entry.copy())

        self.tokenize()
        self.tensorize()

    def tokenize(self):
        """Tokenizes the captions.
        This will add cap_token in each entry of the datasets.
        """
        for entry in self.entries:
            sentence = entry['caption']
            sentence = sentence.lower()
            sentence = sentence.replace(',', '').replace('.', ' ').replace(';', ' ').replace('(', ' ').replace(')',
                                                                                                               ' ').replace(
                '-', ' ').replace('?', '').replace('\'s', ' \'s').replace('s\'', ' ').replace('n\'t', ' not').replace(
                '"', ' ')
            words = sentence.split()
            words = words[:self.max_length]
            caption = " ".join(words)
            tokens = clip.tokenize(caption).squeeze(0)
            entry['caption'] = caption
            entry['cap_token'] = tokens


    def tensorize(self):
        for entry in self.entries:
            caption = torch.from_numpy(np.array(entry['cap_token']))
            entry['cap_token'] = caption
            label_id = entry['label_id']
            if None!=label_id:
                label_id = torch.from_numpy(np.array(entry['label_id'])).long()
                entry['label_id'] = label_id

    def __getitem__(self, index):
        item = {}
        entry = self.entries[index]
        img_path = entry["img_path"]
        img = Image.open(os.path.join(self.root_dir, img_path))
        if self.transform is not None:
            item['img'] = self.transform(img)
        item['caption'] = entry["cap_token"]
        item['label_id'] = entry["label_id"]
        item['img_path'] = img_path
        return item

    def __len__(self):
        return len(self.entries)

class ADDataset(Dataset):
    def __init__(self, split, dataroot, transform=None, nans=2, max_length=48, class_names=None):
        super(ADDataset, self).__init__()
        self.root_dir = dataroot
        self.split = split
        self.transform = transform
        self.num_ans_candidates = nans
        self.max_length = max_length

        self.data_all = []
        meta_info = json.load(open(f'{self.root_dir}/meta_highshot.json', 'r'))
        meta_info = meta_info[split]

        if class_names:
            self.cls_names = class_names
            for cls in self.cls_names:
                assert cls in list(meta_info.keys()), f"Class {cls} is not in dataset."
        else:
            self.cls_names = list(meta_info.keys())

        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list = self.cls_names

    def __len__(self):
        return len(self.data_all)

    def tokenize(self, caption):
        """Tokenizes the captions.
        This will add cap_token in each entry of the datasets.
        """
        sentence = caption
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('.', ' ').replace(';', ' ').replace('(', ' ').replace(')',
                                                                                                           ' ').replace(
            '-', ' ').replace('?', '').replace('\'s', ' \'s').replace('s\'', ' ').replace('n\'t', ' not').replace(
            '"', ' ')
        words = sentence.split()
        words = words[:self.max_length]
        caption = " ".join(words)
        tokens = clip.tokenize(caption).squeeze(0)
        return tokens

    def __getitem__(self, index):
        item = {}
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root_dir, img_path))
        # transforms
        img = self.transform(img) if self.transform is not None else img

        caption = specie_name
        cap_token = self.tokenize(caption)

        item['img'] = img
        item['caption'] = cap_token
        item['label_id'] = anomaly
        item['img_path'] = img_path
        return item



