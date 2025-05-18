import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.entity_predictor import EntityPredictor, compute_entity_loss


class R2GenMultiTaskModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenMultiTaskModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)
        
        # Visual feature extractor
        self.visual_extractor = VisualExtractor(args)
        
        # Report generation decoder
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        
        # Entity prediction head
        self.entity_predictor = EntityPredictor(
            visual_feat_size=args.d_vf,  # Should match the output dimension of VisualExtractor
            vocab_size=self.vocab_size,
            hidden_size=args.d_model,
            dropout=args.dropout,
            dataset_name=args.dataset_name
        )
        
        # Set forward method based on dataset
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, entity_targets=None, mode='train', task=None):
        # Extract visual features from the two images
        # att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        # att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        
        # # Concatenate features from both images
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        # att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        
        # Determine the output based on mode and task
        if mode == 'train':
            # Perform specific task or both tasks
            if task == 'entity':
                if self.args.freeze_visual_extractor_on_task1:
                    with torch.no_grad():
                        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
                        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
                else:
                    att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
                    att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])  
                fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                # Only perform entity prediction
                entity_logits = self.entity_predictor(fc_feats)
                return entity_logits
            elif task == 'report':
                # Only generate report
                if self.args.freeze_visual_extractor_on_task2:
                    with torch.no_grad():
                        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
                        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
                else:
                    att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
                    att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

                with torch.no_grad():
                    fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                    att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
                    entity_logits = self.entity_predictor(fc_feats) #entity predictor 不更新
                    entity_probs = torch.sigmoid(entity_logits)

                    # 使用广播得到相同维度的entity_probs
                    entity_probs_expanded = entity_probs.unsqueeze(1)  # [batch, 1, vocab_size]
                    entity_probs_expanded = entity_probs_expanded.expand(-1, att_feats.size(1), -1)
                    # 现在在乘法操作时会自动广播到 [batch, patch_num, vocab_size]

                    # 创建连接并交互
                    features_a_concat_b = torch.cat((att_feats, entity_probs_expanded), dim=2)  # [batch, patch_num, d_vf+vocab_size]
                    features_b_concat_a = torch.cat((entity_probs_expanded, att_feats), dim=2)  # [batch, patch_num, vocab_size+d_vf]

                    # 对应元素相乘，这里会自动广播entity_probs
                    cross_features = features_a_concat_b * features_b_concat_a
                output = self.encoder_decoder(fc_feats, cross_features, targets, mode='forward') #只更新encoderdecoder
                return output
            else:
                raise ValueError(f"Did not specify task")
                
        elif mode == 'sample':
            # Sampling mode (for evaluation or inference)
            if task == 'entity':
                att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
                att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])   
                fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                # Use concatenated features for both images
                entity_logits = self.entity_predictor(fc_feats)
                return entity_logits
            else:
                # Default to report generation for sampling
                
                with torch.no_grad():
                    att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0]) #resnet 不更新
                    att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])   
                    fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                    att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
                    entity_logits = self.entity_predictor(fc_feats) #entity predictor 不更新
                    entity_probs = torch.sigmoid(entity_logits)

                    # 使用广播得到相同维度的entity_probs
                    entity_probs_expanded = entity_probs.unsqueeze(1)  # [batch, 1, vocab_size]
                    entity_probs_expanded = entity_probs_expanded.expand(-1, att_feats.size(1), -1)

                    # 现在在乘法操作时会自动广播到 [batch, patch_num, vocab_size]

                    # 创建连接并交互
                    features_a_concat_b = torch.cat((att_feats, entity_probs_expanded), dim=2)  # [batch, patch_num, d_vf+vocab_size]
                    features_b_concat_a = torch.cat((entity_probs_expanded, att_feats), dim=2)  # [batch, patch_num, vocab_size+d_vf]

                    # 对应元素相乘，这里会自动广播entity_probs
                    cross_features = features_a_concat_b * features_b_concat_a               
                output, _ = self.encoder_decoder(fc_feats,cross_features , mode='sample')
                return output
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward_mimic_cxr(self, images, targets=None, entity_targets=None, mode='train', task=None):
        # Extract features from a single image
        # att_feats, fc_feats = self.visual_extractor(images)
        
        # Determine the output based on mode and task
        if mode == 'train':
            # Perform specific task or both tasks
            if task == 'entity':
                # Only perform entity prediction
                att_feats, fc_feats = self.visual_extractor(images)
                entity_logits = self.entity_predictor(fc_feats)
                return entity_logits
            elif task == 'report':
                # Only generate report
                with torch.no_grad():
                    att_feats, fc_feats = self.visual_extractor(images)
                    entity_logits = self.entity_predictor(fc_feats)  
                    entity_probs = torch.sigmoid(entity_logits)
                     # 使用广播得到相同维度的entity_probs
                    entity_probs_expanded = entity_probs.unsqueeze(1)  # [batch, 1, vocab_size]
                    entity_probs_expanded = entity_probs_expanded.expand(-1, att_feats.size(1), -1)
                    #

                    features_a_concat_b = torch.cat((att_feats, entity_probs_expanded), dim=2)  # [batch, patch_num, d_vf+vocab_size]
                    features_b_concat_a = torch.cat((entity_probs_expanded, att_feats), dim=2)  # [batch, patch_num, vocab_size+d_vf]

                    # 对应元素相乘，这里会自动广播entity_probs
                    cross_features = features_a_concat_b * features_b_concat_a              
                
                output = self.encoder_decoder(fc_feats, cross_features, targets, mode='forward')
                return output
            else:
                raise ValueError(f"Did not specify task")
                
        elif mode == 'sample':
            # Sampling mode (for evaluation or inference)
            if task == 'entity':
                
                att_feats, fc_feats = self.visual_extractor(images)
                entity_logits = self.entity_predictor(fc_feats)
                return entity_logits
            else:
                # Default to report generation for sampling
                
                with torch.no_grad():
                    att_feats, fc_feats = self.visual_extractor(images)
                    entity_logits = self.entity_predictor(fc_feats)  
                    entity_probs = torch.sigmoid(entity_logits)
                    entity_probs_expanded = entity_probs.unsqueeze(1)  # [batch, 1, vocab_size]
                    entity_probs_expanded = entity_probs_expanded.expand(-1, att_feats.size(1), -1)

                    features_a_concat_b = torch.cat((att_feats, entity_probs_expanded), dim=2)  # [batch, patch_num, d_vf+vocab_size]
                    features_b_concat_a = torch.cat((entity_probs_expanded, att_feats), dim=2)  # [batch, patch_num, vocab_size+d_vf]

                    # 对应元素相乘，这里会自动广播entity_probs
                    cross_features = features_a_concat_b * features_b_concat_a              
                
                output,_ = self.encoder_decoder(fc_feats, cross_features, targets, mode='sample')
                # output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
                return output
        else:
            raise ValueError(f"Unsupported mode: {mode}")