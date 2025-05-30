import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, seed=None, curriculum_ratio=1.0, difficulty_scores=None):
        # Set random seed if provided
        if seed is not None and shuffle:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        vocab_size = tokenizer.get_vocab_size()
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        # If difficulty_scores are provided, set difficulty
        if difficulty_scores is not None:
            self.dataset.set_difficulty_scores(difficulty_scores)

        # curriculum_ratio < 1 enables curriculum learning
        if curriculum_ratio < 1.0 and split == 'train':
            self.dataset.sort_by_difficulty()
            self.dataset.filter_by_curriculum_ratio(curriculum_ratio)

        # Save vocabulary size as class variable
        R2DataLoader.vocab_size = tokenizer.get_vocab_size()
        # print(R2DataLoader.vocab_size)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths, entity_ids, entity_nums = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        
        # Use class variable vocabulary size
        vocab_size = R2DataLoader.vocab_size
        
        # Create multi-hot vector representation
        batch_entity_multihot = np.zeros((len(entity_ids), vocab_size), dtype=float)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks
        
        # Convert entity_ids to multi-hot vector. Our tokenizer currently doesn't have token 0, so entity_id needs to be -1 when computing loss
        for i, entities in enumerate(entity_ids):
            for entity_id in entities:
                if entity_id > 0:  # Avoid padding tokens
                    batch_entity_multihot[i, entity_id-1] = 1.0

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.FloatTensor(batch_entity_multihot)

