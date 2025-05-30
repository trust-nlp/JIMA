import torch
import argparse
import numpy as np
import os
import time
from tqdm import tqdm

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.entity_predictor import compute_entity_loss
from modules.loss import compute_loss
from models.r2gen import R2GenMultiTaskModel

def compute_rank(tensor):
    """Returns the rank (starting from 1) of each element in tensor in descending order"""
    sorted_indices = torch.argsort(tensor, descending=True)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(1, len(tensor) + 1, device=tensor.device)
    return ranks

def compute_entity_difficulty(model, dataloader, tokenizer, device):
    model.eval()
    difficulty_scores = {}

    with torch.no_grad():
        for image_ids, images, _, _, entity_targets in dataloader:
            images = images.to(device)
            entity_targets = entity_targets.to(device)

            logits = model(images, mode='sample', task='entity')  # [B, V]
            probs = torch.sigmoid(logits)  # [B, V]
            vocab_size = probs.size(1)

            for i in range(probs.size(0)):
                true_indices = (entity_targets[i] > 0).nonzero(as_tuple=True)[0]
                if len(true_indices) == 0:
                    difficulty_scores[image_ids[i]] = 0.0
                    continue

                ranks = compute_rank(probs[i])  # [V]
                score = sum((ranks[idx] / vocab_size).item() for idx in true_indices) / len(true_indices)
                difficulty_scores[image_ids[i]] = score
    return difficulty_scores

def compute_report_difficulty(model, dataloader, tokenizer, device):
    model.eval()
    difficulty_scores = {}

    with torch.no_grad():
        for image_ids, images, targets, masks, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            targets_input = targets[:, :-1]
            targets_output = targets[:, 1:]
            masks = masks[:, 1:]

            outputs = model(images, targets_input, mode='train', task='report')  # [B, L-1, V]
            probs = torch.softmax(outputs, dim=-1)
            vocab_size = probs.size(-1)

            for b in range(probs.size(0)):
                total_rank = 0
                valid_len = 0
                for i in range(probs.size(1)):
                    if masks[b, i] == 0:
                        continue
                    gold = targets_output[b, i].item()
                    ranks = compute_rank(probs[b, i])
                    total_rank += (ranks[gold] / vocab_size).item()
                    valid_len += 1
                difficulty_scores[image_ids[b]] = total_rank / max(1, valid_len)
    return difficulty_scores

def update_curriculum_ratio(s_t, s_t_prev, c_prev):
    ratio = 1 - ((s_t - s_t_prev) / (s_t_prev + 1e-8))
    return min(1.0, max(0.1, ratio * c_prev))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze_visual_extractor_on_task2', action='store_true',
                    help='Freeze visual extractor when generating reports (task=report)')
    parser.add_argument('--freeze_visual_extractor_on_task1', action='store_true',
                    help='Freeze visual extractor when')
    parser.add_argument('--joint', action='store_true',
                    help='joint train')
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/project/wli5/JIMA/data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/project/wli5/JIMA/data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # Multi-task settings
    parser.add_argument('--entity_weight', type=float, default=0.5, help='weight for entity prediction loss')
    parser.add_argument('--report_weight', type=float, default=1.0, help='weight for report generation loss')
    parser.add_argument('--use_difficulty', type=bool, default=True, help='whether to use difficulty-aware sampling')
    parser.add_argument('--difficulty_update_freq', type=int, default=5, help='update difficulty every n epochs')
    parser.add_argument('--task_alternating', type=bool, default=False, help='whether to alternate between tasks during training')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Training settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray_multitask', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='random seed.')
    parser.add_argument('--resume', type=str, help='resume from checkpoint.')
    parser.add_argument('--log_period', type=int, default=100, help='log training status every n batches.')
    parser.add_argument('--save_model', type=bool, default=True, help='whether to save checkpoint.')

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()

    # Fix random seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.record_dir, exist_ok=True)
    
    # Create tokenizer
    tokenizer = Tokenizer(args)

    # Create two training dataloaders with different orders
    train_dataloader_report = R2DataLoader(args, tokenizer, split='train', shuffle=True, seed=args.seed)
    train_dataloader_entity = R2DataLoader(args, tokenizer, split='train', shuffle=True, seed=args.seed+1)  # Use different seed
    
    # Validation and test dataloaders
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # Build model
    model = R2GenMultiTaskModel(args, tokenizer)
    model = model.to(device)
    
    # Set DataParallel if using multiple GPUs
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Get loss function
    criterion = compute_loss
    
    # Build optimizer and learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    
    # Set entity loss weight
    entity_loss_weight = args.entity_weight
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_score = float('-inf') if args.monitor_mode == 'max' else float('inf')
    not_improved_count = 0
    
    curriculum_ratio = 1.0
    c_prev = 1.0
    s_t_prev = None  # For storing previous performance
    report_diff = None
    entity_diff = None
    warm_up_round = 10

    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        not_improved_count = checkpoint['not_improved_count']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print(f"Checkpoint loaded. Resume training from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Use curriculum_ratio to build dataloader (rebuild each round)
        # Each round creates dataloader based on current curriculum_ratio
        train_dataloader_report = R2DataLoader(
            args, tokenizer, split='train', shuffle=True,
            seed=args.seed, curriculum_ratio=curriculum_ratio,  # Use task2 difficulty
            difficulty_scores=report_diff if epoch+2 > warm_up_round else None
        )
        train_dataloader_entity = R2DataLoader(
            args, tokenizer, split='train', shuffle=True,
            seed=args.seed + 1, curriculum_ratio=curriculum_ratio,  # Use task1 difficulty
            difficulty_scores=entity_diff if epoch+2 > warm_up_round else None
        )


        # Train one epoch
        if not args.joint:
            print(f'alternating training for epoch {epoch+1}...')
            train_losses = train_epoch_alternating(
                model, 
                train_dataloader_report, 
                train_dataloader_entity, 
                optimizer, 
                criterion, 
                device, 
                entity_loss_weight
            )
        else:
            train_losses = train_epoch_joint(
                model, 
                train_dataloader_report, 
                train_dataloader_entity, 
                optimizer, 
                criterion, 
                device, 
                entity_loss_weight
            )
        
        # Validate
        val_losses, val_metrics = validate(
            model, 
            val_dataloader, 
            criterion, 
            tokenizer, 
            device, 
            entity_loss_weight
        )
        
        ### Curriculum control logic (from 10th round onwards)
        if epoch+1 > warm_up_round:
            print("Computing difficulty scores for curriculum learning...")

            # Use full training set to calculate difficulty
            full_train_loader = R2DataLoader(args, tokenizer, split='train', shuffle=False, seed=args.seed)
            entity_diff = compute_entity_difficulty(model, full_train_loader, tokenizer, device)
            report_diff = compute_report_difficulty(model, full_train_loader, tokenizer, device)

            # Current performance (average BLEU or entity_f1)
            s_t = val_metrics.get(args.monitor_metric, 0)

            # Update curriculum_ratio
            if s_t_prev is not None:
                curriculum_ratio = update_curriculum_ratio(s_t, s_t_prev, c_prev)
                c_prev = curriculum_ratio
            s_t_prev = s_t

            print(f"Updated curriculum ratio: {curriculum_ratio:.4f}")

        # Learning rate scheduler
        lr_scheduler.step()
        
        # Record log
        print(f"Epoch {epoch+1} - Train Loss: {train_losses['total_loss']:.4f}, Val Loss: {val_losses['total_loss']:.4f}")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Check if need to save model
        if args.monitor_mode == 'max':
            is_improved = (val_metrics.get(args.monitor_metric, 0) > best_score)
        else:
            is_improved = (val_metrics.get(args.monitor_metric, float('inf')) < best_score)
        
        if is_improved:
            best_score = val_metrics.get(args.monitor_metric, best_score)
            not_improved_count = 0
            
            # Save best model
            state_dict = model.module.state_dict() if args.n_gpu > 1 else model.state_dict()
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_score': best_score,
                'not_improved_count': not_improved_count
            }, os.path.join(args.save_dir, 'model_best.pth'))
            print(f"Saving best model with {args.monitor_metric}: {best_score:.4f}")
        else:
            not_improved_count += 1
        
        # Periodically save model
        if (epoch + 1) % args.save_period == 0:
            state_dict = model.module.state_dict() if args.n_gpu > 1 else model.state_dict()
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_score': best_score,
                'not_improved_count': not_improved_count
            }, os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))
        
        # Early stop
        if not_improved_count > args.early_stop:
            print(f"Validation performance didn\'t improve for {args.early_stop} epochs. Training stops.")
            break
    
    # Test best model
    print("Loading best model for testing...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'model_best.pth'))
    if args.n_gpu > 1:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    
    test_loss, test_metrics = validate(model, test_dataloader, criterion, tokenizer, device, entity_loss_weight)
    print("Test Results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("Training completed!")


def train_epoch_alternating(model, train_dataloader_report, train_dataloader_entity, optimizer, criterion, device, entity_loss_weight):
    """Train one epoch by alternating between report generation and entity prediction tasks"""
    model.train()
    
    total_report_loss = 0
    total_entity_loss = 0
    total_loss = 0
    report_batches = 0
    entity_batches = 0
    
    # Get iterators for both tasks
    report_iter = iter(train_dataloader_report)
    entity_iter = iter(train_dataloader_entity)
    
    # Track if data loaders are exhausted
    report_exhausted = False
    entity_exhausted = False
    
    # Calculate length of longer dataloader as baseline
    max_batches = max(len(train_dataloader_report), len(train_dataloader_entity))
    report_size = len(train_dataloader_report)
    entity_size = len(train_dataloader_entity)
    
    print(f"Report batches: {report_size}, Entity batches: {entity_size}")
    
    # Alternate between tasks, ensuring complete training for both
    batch_idx = 0
    while batch_idx < 2 * max_batches:  # Ensure loop is long enough to process all data
        # End loop if both tasks have exhausted their data
        if report_exhausted and entity_exhausted:
            break
            
        # Process report generation task on even batches
        if batch_idx % 2 == 0 and not report_exhausted:
            try:
                # Report generation task
                images_id, images, reports_ids, reports_masks, _ = next(report_iter)
                images = images.to(device)
                reports_ids = reports_ids.to(device)
                reports_masks = reports_masks.to(device)
                
                # Forward pass
                output = model(images, reports_ids, mode='train', task='report')
                
                # Calculate loss
                report_loss = criterion(output, reports_ids, reports_masks)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                report_loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                
                # Accumulate loss
                total_report_loss += report_loss.item()
                total_loss += report_loss.item()
                report_batches += 1
                
                # Output progress
                if report_batches % 20 == 0:
                    print(f"Report Batch [{report_batches}/{report_size}] - Loss: {report_loss.item():.4f}")
                    
            except StopIteration:
                # Mark report generation data as exhausted
                report_exhausted = True
                print("Report task data exhausted.")
        
        # Process entity prediction task on odd batches
        elif batch_idx % 2 == 1 and not entity_exhausted:
            try:
                # Entity prediction task
                images_id, images, _, _, entity_targets = next(entity_iter)
                images = images.to(device)
                entity_targets = entity_targets.to(device)
                
                # Forward pass
                entity_logits = model(images, mode='train', task='entity')
                
                # Calculate loss
                entity_loss = compute_entity_loss(entity_logits, entity_targets)
                weighted_entity_loss = entity_loss_weight * entity_loss
                
                # Backward pass and optimization
                optimizer.zero_grad()
                weighted_entity_loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                
                # Accumulate loss
                total_entity_loss += entity_loss.item()
                total_loss += weighted_entity_loss.item()
                entity_batches += 1
                
                # Output progress
                if entity_batches % 20 == 0:
                    print(f"Entity Batch [{entity_batches}/{entity_size}] - Loss: {entity_loss.item():.4f}")
                    
            except StopIteration:
                # Mark entity prediction data as exhausted
                entity_exhausted = True
                print("Entity task data exhausted.")
        
        # Increment batch counter
        batch_idx += 1
    
    # Calculate average losses
    avg_report_loss = total_report_loss / max(1, report_batches)
    avg_entity_loss = total_entity_loss / max(1, entity_batches)
    avg_total_loss = total_loss / max(1, report_batches + entity_batches)
    
    print(f"Epoch Summary - Report Loss: {avg_report_loss:.4f}, Entity Loss: {avg_entity_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
    
    # Return dictionary with required keys
    return {
        'report_loss': avg_report_loss,
        'entity_loss': avg_entity_loss,
        'total_loss': avg_total_loss
    }

def train_epoch_joint(model, train_dataloader_report, train_dataloader_entity, optimizer, criterion, device, entity_loss_weight):
    """Joint training of both tasks in each batch"""
    model.train()

    total_report_loss = 0
    total_entity_loss = 0
    total_loss = 0
    num_batches = 0

    # Get iterators for both tasks
    report_iter = iter(train_dataloader_report)
    entity_iter = iter(train_dataloader_entity)

    # Track if data loaders are exhausted
    report_exhausted = False
    entity_exhausted = False

    while not (report_exhausted and entity_exhausted):
        optimizer.zero_grad()
        batch_loss = 0
        
        # Process report generation task
        if not report_exhausted:
            try:
                images_id, images, reports_ids, reports_masks, _ = next(report_iter)
                images = images.to(device)
                reports_ids = reports_ids.to(device)
                reports_masks = reports_masks.to(device)
                
                # Forward pass
                output = model(images, reports_ids, mode='train', task='report')
                
                # Calculate loss
                report_loss = criterion(output, reports_ids, reports_masks)
                batch_loss += report_loss
                total_report_loss += report_loss.item()
                
            except StopIteration:
                report_exhausted = True
        
        # Process entity prediction task
        if not entity_exhausted:
            try:
                images_id, images, _, _, entity_targets = next(entity_iter)
                images = images.to(device)
                entity_targets = entity_targets.to(device)
                
                # Forward pass
                entity_logits = model(images, mode='train', task='entity')
                
                # Calculate loss
                entity_loss = compute_entity_loss(entity_logits, entity_targets)
                weighted_entity_loss = entity_loss_weight * entity_loss
                batch_loss += weighted_entity_loss
                total_entity_loss += entity_loss.item()
                
            except StopIteration:
                entity_exhausted = True
        
        # Backward pass and optimization
        if batch_loss > 0:
            batch_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item()
            num_batches += 1
        
        # Output progress
        if num_batches % 20 == 0:
            print(f"Batch [{num_batches}] - Report Loss: {total_report_loss/max(1,num_batches):.4f}, "
                    f"Entity Loss: {total_entity_loss/max(1,num_batches):.4f}, "
                    f"Total Loss: {total_loss/max(1,num_batches):.4f}")

    # Calculate average losses
    avg_report_loss = total_report_loss / max(1, num_batches)
    avg_entity_loss = total_entity_loss / max(1, num_batches)
    avg_total_loss = total_loss / max(1, num_batches)

    return {
        'report_loss': avg_report_loss,
        'entity_loss': avg_entity_loss,
        'total_loss': avg_total_loss
    }
        
               
        
def validate(model, dataloader, criterion, tokenizer, device, entity_loss_weight):
    """Validate model performance"""
    model.eval()
    
    total_report_loss = 0
    total_entity_loss = 0
    total_loss = 0
    
    all_reports = []
    all_ground_truths = []
    all_entity_logits = []
    all_entity_targets = []
    
    with torch.no_grad():
        for batch_idx, (images_id, images, reports_ids, reports_masks, entity_targets) in enumerate(dataloader):
            images = images.to(device)
            reports_ids = reports_ids.to(device)
            reports_masks = reports_masks.to(device)
            entity_targets = entity_targets.to(device)
            
            # Generate reports
            output = model(images, mode='sample')
            reports = tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            
            all_reports.extend(reports)
            all_ground_truths.extend(ground_truths)
            
            # Entity prediction
            entity_logits = model(images, mode='sample', task='entity')
            all_entity_logits.append(entity_logits.cpu())
            all_entity_targets.append(entity_targets.cpu())
            
            # Calculate report generation loss (using training mode for loss calculation)
            output_for_loss = model(images, reports_ids, mode='train', task='report')
            report_loss = criterion(output_for_loss, reports_ids, reports_masks)
            
            # Calculate entity prediction loss
            entity_loss = compute_entity_loss(entity_logits, entity_targets)
            
            # Calculate total loss
            total_loss_batch = report_loss + entity_loss_weight * entity_loss
            
            # Accumulate losses
            total_report_loss += report_loss.item()
            total_entity_loss += entity_loss.item()
            total_loss += total_loss_batch.item()
    
    # Calculate average losses
    avg_report_loss = total_report_loss / len(dataloader)
    avg_entity_loss = total_entity_loss / len(dataloader)
    avg_total_loss = total_loss / len(dataloader)
    
    # Calculate report generation metrics
    reports_metrics = compute_scores({i: [gt] for i, gt in enumerate(all_ground_truths)},
                                   {i: [re] for i, re in enumerate(all_reports)})
    
    # Calculate entity prediction metrics
    all_entity_logits = torch.cat(all_entity_logits, dim=0)
    all_entity_targets = torch.cat(all_entity_targets, dim=0)
    entity_preds = (torch.sigmoid(all_entity_logits) > 0.5).float()
    
    # Calculate simple F1 score
    tp = (entity_preds * all_entity_targets).sum(dim=0)
    fp = (entity_preds * (1 - all_entity_targets)).sum(dim=0)
    fn = ((1 - entity_preds) * all_entity_targets).sum(dim=0)
    
    precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
    recall = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Add entity prediction metrics
    entity_metrics = {
        'entity_precision': precision.item(),
        'entity_recall': recall.item(),
        'entity_f1': f1.item()
    }
    
    # Merge metrics
    metrics = {**reports_metrics, **entity_metrics}
    
    return {
        'report_loss': avg_report_loss,
        'entity_loss': avg_entity_loss,
        'total_loss': avg_total_loss
    }, metrics


if __name__ == '__main__':
    main()