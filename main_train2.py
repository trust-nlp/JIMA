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
from models.r2gen2 import R2GenMultiTaskModel

def compute_rank(tensor):
    """返回 tensor 中每个元素在降序排列中的排名（从 1 开始）"""
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
        for image_ids, images, targets, masks, entity_targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            targets_input = targets[:, :-1]
            targets_output = targets[:, 1:]
            masks = masks[:, 1:]

            outputs = model(images, targets_input, entity_targets, mode='train', task='report')  # [B, L-1, V]
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
    # 解析参数
    args = parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.record_dir, exist_ok=True)
    
    # 创建分词器
    tokenizer = Tokenizer(args)

    # 创建两个不同顺序的训练数据加载器
    train_dataloader_report = R2DataLoader(args, tokenizer, split='train', shuffle=True, seed=args.seed)
    # train_dataloader_entity = R2DataLoader(args, tokenizer, split='train', shuffle=True, seed=args.seed+1)  # 使用不同的种子
    
    # 验证和测试数据加载器
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # 构建模型
    model = R2GenMultiTaskModel(args, tokenizer)
    model = model.to(device)
    
    # 如果使用多GPU，设置DataParallel
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # 获取损失函数
    criterion = compute_loss
    
    # 构建优化器和学习率调度器
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    
    # 设置实体损失权重
    entity_loss_weight = args.entity_weight
    
    # 如果指定了resume，从检查点恢复
    start_epoch = 0
    best_score = float('-inf') if args.monitor_mode == 'max' else float('inf')
    not_improved_count = 0
    
    curriculum_ratio = 1.0
    c_prev = 1.0
    s_t_prev = None  # 用于保存上一轮性能
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
    
    # 训练循环
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # 使用 curriculum_ratio 构建 dataloader（每轮重新构建）
        # 每一轮根据当前 curriculum_ratio 创建 dataloader
        train_dataloader_report = R2DataLoader(
            args, tokenizer, split='train', shuffle=True,
            seed=args.seed, curriculum_ratio=curriculum_ratio,  # 用 task2 难度
            difficulty_scores=report_diff if epoch+2 > warm_up_round else None
        )
        # train_dataloader_entity = R2DataLoader(
        #     args, tokenizer, split='train', shuffle=True,
        #     seed=args.seed + 1, curriculum_ratio=curriculum_ratio,  # 用 task1 难度
        #     difficulty_scores=entity_diff if epoch+2 > warm_up_round else None
        # )


        # 训练一个epoch
        # train_losses = train_epoch_alternating(
        #     model, 
        #     train_dataloader_report, 
        #     train_dataloader_entity, 
        #     optimizer, 
        #     criterion, 
        #     device, 
        #     entity_loss_weight
        # )
        train_losses = train_epoch_joint(
            model, 
            train_dataloader_report, 
            optimizer, 
            criterion, 
            device, 
            entity_loss_weight
        )
        
        # 验证
        val_losses, val_metrics = validate(
            model, 
            val_dataloader, 
            criterion, 
            tokenizer, 
            device, 
            entity_loss_weight
        )
        
        ### Curriculum 控制逻辑（从第10轮开始）
        if epoch+1 > warm_up_round:
            print("Computing difficulty scores for curriculum learning...")

            # 用完整训练集计算 difficulty
            full_train_loader = R2DataLoader(args, tokenizer, split='train', shuffle=False, seed=args.seed)
            # entity_diff = compute_entity_difficulty(model, full_train_loader, tokenizer, device)
            report_diff = compute_report_difficulty(model, full_train_loader, tokenizer, device)

            # 当前性能（平均 BLEU 或 entity_f1）
            s_t = val_metrics.get(args.monitor_metric, 0)

            # 更新 curriculum_ratio
            if s_t_prev is not None:
                curriculum_ratio = update_curriculum_ratio(s_t, s_t_prev, c_prev)
                c_prev = curriculum_ratio
            s_t_prev = s_t

            print(f"Updated curriculum ratio: {curriculum_ratio:.4f}")

        # 学习率调度
        lr_scheduler.step()
        
        # 记录日志
        print(f"Epoch {epoch+1} - Train Loss: {train_losses['total_loss']:.4f}, Val Loss: {val_losses['total_loss']:.4f}")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # 检查是否需要保存模型
        if args.monitor_mode == 'max':
            is_improved = (val_metrics.get(args.monitor_metric, 0) > best_score)
        else:
            is_improved = (val_metrics.get(args.monitor_metric, float('inf')) < best_score)
        
        if is_improved:
            best_score = val_metrics.get(args.monitor_metric, best_score)
            not_improved_count = 0
            
            # 保存最佳模型
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
        
        # 定期保存模型
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
        
        # 早停
        if not_improved_count > args.early_stop:
            print(f"Validation performance didn\'t improve for {args.early_stop} epochs. Training stops.")
            break
    
    # 测试最佳模型
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
    """训练一个epoch，交替使用报告生成和实体预测任务的批次"""
    model.train()
    
    total_report_loss = 0
    total_entity_loss = 0
    total_loss = 0
    report_batches = 0
    entity_batches = 0
    
    # 获取两个任务的迭代器
    report_iter = iter(train_dataloader_report)
    entity_iter = iter(train_dataloader_entity)
    
    # 记录两个数据加载器是否已用完
    report_exhausted = False
    entity_exhausted = False
    
    # 计算较长的数据加载器的长度作为基准
    max_batches = max(len(train_dataloader_report), len(train_dataloader_entity))
    report_size = len(train_dataloader_report)
    entity_size = len(train_dataloader_entity)
    
    print(f"Report batches: {report_size}, Entity batches: {entity_size}")
    
    # 交替使用两个任务的批次，确保两个任务都能完整训练
    batch_idx = 0
    while batch_idx < 2 * max_batches:  # 确保循环足够长以处理所有数据
        # 如果两个任务都已经用完数据，则结束循环
        if report_exhausted and entity_exhausted:
            break
            
        # 偶数批次处理报告生成任务
        if batch_idx % 2 == 0 and not report_exhausted:
            try:
                # 报告生成任务
                images_id, images, reports_ids, reports_masks, _ = next(report_iter)
                images = images.to(device)
                reports_ids = reports_ids.to(device)
                reports_masks = reports_masks.to(device)
                
                # 前向传播
                output = model(images, reports_ids, mode='train', task='report')
                
                # 计算损失
                report_loss = criterion(output, reports_ids, reports_masks)
                
                # 反向传播和优化
                optimizer.zero_grad()
                report_loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                
                # 累计损失
                total_report_loss += report_loss.item()
                total_loss += report_loss.item()
                report_batches += 1
                
                # 输出进度
                if report_batches % 20 == 0:
                    print(f"Report Batch [{report_batches}/{report_size}] - Loss: {report_loss.item():.4f}")
                    
            except StopIteration:
                # 如果报告生成数据已经用完，标记为耗尽
                report_exhausted = True
                print("Report task data exhausted.")
        
        # 奇数批次处理实体预测任务
        elif batch_idx % 2 == 1 and not entity_exhausted:
            try:
                # 实体预测任务
                images_id, images, _, _, entity_targets = next(entity_iter)
                images = images.to(device)
                entity_targets = entity_targets.to(device)
                
                # 前向传播
                entity_logits = model(images, mode='train', task='entity')
                
                # 计算损失
                entity_loss = compute_entity_loss(entity_logits, entity_targets)
                weighted_entity_loss = entity_loss_weight * entity_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                weighted_entity_loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                
                # 累计损失
                total_entity_loss += entity_loss.item()
                total_loss += weighted_entity_loss.item()
                entity_batches += 1
                
                # 输出进度
                if entity_batches % 20 == 0:
                    print(f"Entity Batch [{entity_batches}/{entity_size}] - Loss: {entity_loss.item():.4f}")
                    
            except StopIteration:
                # 如果实体预测数据已经用完，标记为耗尽
                entity_exhausted = True
                print("Entity task data exhausted.")
        
        # 增加批次计数
        batch_idx += 1
    
    # 计算平均损失
    avg_report_loss = total_report_loss / max(1, report_batches)
    avg_entity_loss = total_entity_loss / max(1, entity_batches)
    avg_total_loss = total_loss / max(1, report_batches + entity_batches)
    
    print(f"Epoch Summary - Report Loss: {avg_report_loss:.4f}, Entity Loss: {avg_entity_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
    
    # 确保返回字典含有所需的键
    return {
        'report_loss': avg_report_loss,
        'entity_loss': avg_entity_loss,
        'total_loss': avg_total_loss
    }

def train_epoch_joint(model, train_dataloader_report, optimizer, criterion, device, entity_loss_weight):
    """Joint training of report generation only (entity prediction removed)"""
    model.train()

    total_report_loss = 0
    total_loss = 0
    num_batches = 0

    report_iter = iter(train_dataloader_report)
    report_exhausted = False

    while not report_exhausted:
        optimizer.zero_grad()
        batch_loss = 0

        try:
            images_id, images, reports_ids, reports_masks, entity_targets = next(report_iter)
            images = images.to(device)
            reports_ids = reports_ids.to(device)
            reports_masks = reports_masks.to(device)
            entity_targets = entity_targets.to(device)

            output = model(images, reports_ids, entity_targets=entity_targets, mode='train', task='report')
            report_loss = criterion(output, reports_ids, reports_masks)

            batch_loss += report_loss
            total_report_loss += report_loss.item()
        except StopIteration:
            report_exhausted = True

        if batch_loss > 0:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item()
            num_batches += 1

        if num_batches % 20 == 0:
            print(f"Batch [{num_batches}] - Report Loss: {total_report_loss/max(1,num_batches):.4f}, Total Loss: {total_loss/max(1,num_batches):.4f}")

    avg_report_loss = total_report_loss / max(1, num_batches)
    avg_total_loss = total_loss / max(1, num_batches)

    return {
        'report_loss': avg_report_loss,
        'total_loss': avg_total_loss
    }
        
               
        
def validate(model, dataloader, criterion, tokenizer, device, entity_loss_weight):
    """验证模型性能"""
    model.eval()

    total_report_loss = 0
    total_entity_loss = 0
    total_loss = 0

    all_reports = []
    all_ground_truths = []

    with torch.no_grad():
        for images_id, images, reports_ids, reports_masks, entity_targets in dataloader:
            images = images.to(device)
            reports_ids = reports_ids.to(device)
            reports_masks = reports_masks.to(device)
            entity_targets = entity_targets.to(device)

            output = model(images, mode='sample')
            reports = tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

            all_reports.extend(reports)
            all_ground_truths.extend(ground_truths)

            output_for_loss = model(images, reports_ids, entity_targets=entity_targets, mode='train', task='report')
            report_loss = criterion(output_for_loss, reports_ids, reports_masks)

            total_report_loss += report_loss.item()
            total_loss += report_loss.item()

    avg_report_loss = total_report_loss / len(dataloader)
    avg_total_loss = total_loss / len(dataloader)

    reports_metrics = compute_scores({i: [gt] for i, gt in enumerate(all_ground_truths)},
                                     {i: [re] for i, re in enumerate(all_reports)})

    metrics = {**reports_metrics}

    return {
        'report_loss': avg_report_loss,
        'total_loss': avg_total_loss
    }, metrics


if __name__ == '__main__':
    main()