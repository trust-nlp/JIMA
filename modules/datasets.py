import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]

        for i in range(len(self.examples)):                    
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            self.examples[i]['entity_ids'] = [tokenizer.get_id_by_token(token) for token in self.examples[i]['tokens']]
            
            self.examples[i]['entity_nums'] = len(self.examples[i]['entity_ids'])
            self.examples[i]['difficulty'] = 0.0
    def set_difficulty_scores(self, difficulty_dict):
        """
        外部接口：设置每个样本的 difficulty 分数（由 image_id 索引）
        """
        for example in self.examples:
            image_id = example['id']
            if image_id in difficulty_dict:
                example['difficulty'] = difficulty_dict[image_id]

    def sort_by_difficulty(self):
        """
        将样本按 difficulty 升序排序（越容易排在越前面）
        """
        self.examples.sort(key=lambda x: x['difficulty'])

    def filter_by_curriculum_ratio(self, ratio):
        """
        保留前 ratio 百分比的样本
        """
        num = int(len(self.examples) * ratio)
        self.examples = self.examples[:max(1, num)]
    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        # print(example)
        # exit()
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        entity_ids = example['entity_ids']
        entity_nums = example['entity_nums']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, entity_ids,entity_nums)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        entity_ids = example['entity_ids']
        entity_nums = example['entity_nums']
        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length, entity_ids, entity_nums)
        return sample
