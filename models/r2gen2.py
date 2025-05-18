import torch
import torch.nn as nn
import numpy as np
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.entity_predictor import EntityPredictor

class R2GenMultiTaskModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenMultiTaskModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)

        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.entity_predictor = EntityPredictor(
            visual_feat_size=args.d_vf,
            vocab_size=self.vocab_size,
            hidden_size=args.d_model,
            dropout=args.dropout,
            dataset_name=args.dataset_name
        )

        self.forward = self.forward_iu_xray if args.dataset_name == 'iu_xray' else self.forward_mimic_cxr

    def forward_iu_xray(self, images, targets=None, entity_targets=None, mode='train', task=None):
        if mode == 'train':
            if task == 'report':
                with torch.no_grad():
                    att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
                    att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
                    fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                    att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
                    # entity_probs = entity_targets
                    entity_probs = entity_targets.to(att_feats.device)
                    entity_probs_expanded = entity_probs.unsqueeze(1).expand(-1, att_feats.size(1), -1)
                    features_a_concat_b = torch.cat((att_feats, entity_probs_expanded), dim=2)
                    features_b_concat_a = torch.cat((entity_probs_expanded, att_feats), dim=2)
                    cross_features = features_a_concat_b * features_b_concat_a
                return self.encoder_decoder(fc_feats, cross_features, targets, mode='forward')
            else:
                raise ValueError("Task must be 'report'")

        elif mode == 'sample':
            with torch.no_grad():
                att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
                att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
                fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
                entity_logits = self.entity_predictor(fc_feats)
                entity_probs = torch.sigmoid(entity_logits)
                entity_probs_expanded = entity_probs.unsqueeze(1).expand(-1, att_feats.size(1), -1)
                features_a_concat_b = torch.cat((att_feats, entity_probs_expanded), dim=2)
                features_b_concat_a = torch.cat((entity_probs_expanded, att_feats), dim=2)
                cross_features = features_a_concat_b * features_b_concat_a
            output, _ = self.encoder_decoder(fc_feats, cross_features, mode='sample')
            return output
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward_mimic_cxr(self, images, targets=None, entity_targets=None, mode='train', task=None):
        if mode == 'train':
            if task == 'report':
                with torch.no_grad():
                    att_feats, fc_feats = self.visual_extractor(images)
                    # entity_probs = entity_targets
                    entity_probs = entity_targets.to(att_feats.device)
                    entity_probs_expanded = entity_probs.unsqueeze(1).expand(-1, att_feats.size(1), -1)
                    features_a_concat_b = torch.cat((att_feats, entity_probs_expanded), dim=2)
                    features_b_concat_a = torch.cat((entity_probs_expanded, att_feats), dim=2)
                    cross_features = features_a_concat_b * features_b_concat_a
                return self.encoder_decoder(fc_feats, cross_features, targets, mode='forward')
            else:
                raise ValueError("Task must be 'report'")

        elif mode == 'sample':
            with torch.no_grad():
                att_feats, fc_feats = self.visual_extractor(images)
                entity_logits = self.entity_predictor(fc_feats)
                entity_probs = torch.sigmoid(entity_logits)
                entity_probs_expanded = entity_probs.unsqueeze(1).expand(-1, att_feats.size(1), -1)
                features_a_concat_b = torch.cat((att_feats, entity_probs_expanded), dim=2)
                features_b_concat_a = torch.cat((entity_probs_expanded, att_feats), dim=2)
                cross_features = features_a_concat_b * features_b_concat_a
            output, _ = self.encoder_decoder(fc_feats, cross_features, mode='sample')
            return output
        else:
            raise ValueError(f"Unsupported mode: {mode}")
