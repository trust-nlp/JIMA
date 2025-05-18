import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityPredictor(nn.Module):
    """
    Predicts medical entities from visual features.
    Input: Image features (fc_feats) from the visual extractor
           For IU X-ray, these are concatenated features from two views
    Output: Entity predictions in a multi-label classification setup
    """
    def __init__(self, visual_feat_size, vocab_size, hidden_size=512, dropout=0.1, dataset_name='iu_xray'):
        super(EntityPredictor, self).__init__()
        
        # For IU X-ray, the input features are concatenated from two views,
        # so the input dimension is doubled
        self.dataset_name = dataset_name
        input_dim = visual_feat_size * 2 if dataset_name == 'iu_xray' else visual_feat_size
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )
        
    def forward(self, avg_feats):
        """
        Args:
            avg_feats: Average visual features from visual extractor (batch_size, visual_feat_size)
        Returns:
            entity_logits: Logits for entity prediction (batch_size, vocab_size)
        """
        entity_logits = self.projection(avg_feats)
        return entity_logits


def compute_entity_loss(entity_logits, entity_targets):
    """
    Compute multi-label classification loss for entity prediction
    
    Args:
        entity_logits: Logits from entity predictor (batch_size, vocab_size)
        entity_targets: Ground truth entity labels as multi-hot vector (batch_size, vocab_size)
        
    Returns:
        loss: Binary cross entropy loss with logits
    """
    loss = F.binary_cross_entropy_with_logits(entity_logits, entity_targets)
    return loss