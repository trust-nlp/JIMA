import json
import re
from collections import Counter
import argparse
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from radgraph import RadGraph
from concurrent.futures import ThreadPoolExecutor, as_completed

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

def process_report(report_text, radgraph_model):
    """Process a single report with RadGraph and extract tokens, splitting multi-word tokens"""
    annotations = radgraph_model([report_text])
    
    # Extract tokens and split multi-word tokens
    tokens = []
    if annotations and '0' in annotations:
        for entity_id, entity_data in annotations['0']['entities'].items():
            # Split multi-word tokens by space
            token_text = entity_data['tokens']
            individual_tokens = token_text.split()
            tokens.extend(individual_tokens)
    
    return tokens

def process_dataset_with_radgraph(dataset_dict, num_workers=8):
    # Initialize RadGraph
    radgraph = RadGraph()
    
    # Create a new dictionary to store the processed data
    processed_data = {}
    
    # Process each split
    for split_name, split_dataset in dataset_dict.items():
        print(f"Processing {split_name} split...")
        processed_split = []
        all_samples = list(split_dataset)
        
        # Create a progress bar
        pbar = tqdm(total=len(all_samples), desc=f"Processing {split_name}")
        
        # Setup ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks
            future_to_sample = {}
            for sample in all_samples:
                future = executor.submit(process_report, sample['report'], radgraph)
                future_to_sample[future] = sample
            
            # Process results as they complete
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    tokens = future.result()
                    new_sample = sample.copy()
                    new_sample['tokens'] = tokens
                    processed_split.append(new_sample)
                except Exception as exc:
                    print(f"Sample processing generated an exception: {exc}")
                    # Still add the sample but with empty tokens
                    new_sample = sample.copy()
                    new_sample['tokens'] = []
                    processed_split.append(new_sample)
                
                # Update progress bar
                pbar.update(1)
        
        # Close progress bar
        pbar.close()
        processed_data[split_name] = processed_split
    
    return processed_data



def main():
    parser = argparse.ArgumentParser(description='Complete data processing pipeline: RadGraph + Token filtering')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input annotation JSON file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to final output JSON file')
    parser.add_argument('--threshold', type=int, default=3, help='Token frequency threshold')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['iu_xray', 'mimic_cxr'], 
                       help='Dataset name for text cleaning')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads for RadGraph processing')
    
    args = parser.parse_args()
    
    print("Loading dataset...")
    with open(args.input_path, 'r') as f:
        raw_data = json.load(f)
    
    # Step 1: Create tokenizer from original data
    print("Creating tokenizer...")
    args.ann_path = args.input_path
    tokenizer = Tokenizer(args)
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Step 2: RadGraph processing + filtering
    print("Processing with RadGraph and filtering tokens...")
    dataset_dict = {
        split: Dataset.from_list(raw_data[split]) for split in raw_data.keys()
    }
    dataset = DatasetDict(dataset_dict)
    
    # Process with RadGraph
    processed_data = process_dataset_with_radgraph(dataset, args.num_workers)
    
    # Filter tokens in place
    total_tokens_before = 0
    total_tokens_after = 0
    
    for split_name, split_data in processed_data.items():
        for sample in split_data:
            if 'tokens' in sample:
                original_tokens = sample['tokens']
                total_tokens_before += len(original_tokens)
                
                lowercase_tokens = [token.lower() for token in original_tokens]
                filtered_tokens = [token for token in lowercase_tokens if token in tokenizer.token2idx]
                
                total_tokens_after += len(filtered_tokens)
                sample['tokens'] = filtered_tokens
    
    print(f"Total tokens before filtering: {total_tokens_before}")
    print(f"Total tokens after filtering: {total_tokens_after}")
    print(f"Removed tokens: {total_tokens_before - total_tokens_after}")
    
    # Save final result
    print(f"Saving final result to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(processed_data, f)
    
    print("Done!")

if __name__ == "__main__":
    main()
