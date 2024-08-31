import torch
from torch import nn

# @LOSS.register('ms_loss')
class MultiSimilarityLoss(nn.Module):
    def __init__(self, scale_pos, scale_neg, iou_threshold, lamdba=0.5, reweight_func='none'):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = lamdba # lambda=0.5
        self.margin = 0.1

        self.scale_pos = scale_pos # scale_pos: alpha
        self.scale_neg = scale_neg # scale_neg: beta
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_func)

    def forward_py(self, feats, labels, ious):
        # feat shape T[1024, 128] labels T[1024,]  ious T[1024,]
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        similarity = torch.matmul(feats, torch.t(feats)) # T[1024, 1024]

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = similarity[i][labels == labels[i]] # find all same labels from ground truth
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon] # 去除自身相似 T[13]
            neg_pair_ = similarity[i][labels != labels[i]] #  T[1010]

            neg_pair = neg_pair_[(neg_pair_ + self.margin) > min(pos_pair_)] # 
            pos_pair = pos_pair_[(pos_pair_ - self.margin) < max(neg_pair_)] # S_ik

            if len(neg_pair) == 0 or len(pos_pair) == 0:
                continue

            # weighting step
            pos_loss = torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)))
            pos_loss = 1 / self.scale_pos * torch.log(1 + pos_loss)
            neg_loss = torch.sum(torch.exp( self.scale_neg * (neg_pair - self.thresh)))
            neg_loss = 1 / self.scale_neg * torch.log(1 + neg_loss)
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size # 1/m
        return loss

    def forward(self, features, labels, ious): # labels 最大值是 bg
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        
        similarity = torch.matmul(features, features.T) 
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        pos_label_mask = torch.eq(labels, labels.T).cuda()  # BFP_SHAPE T(B * 128, 2048) B*128 == 2048
        neg_label_mask = (~pos_label_mask)
        pos_label_mask.fill_diagonal_(False)
        pos_sim_min = torch.min(similarity[pos_label_mask])
        neg_sim_max = torch.max(similarity[neg_label_mask])
        neg_label_mask = neg_label_mask & ((similarity + self.margin) > pos_sim_min)
        pos_label_mask = pos_label_mask & ((similarity - self.margin) < neg_sim_max)
        loss = torch.tensor(0., device="cuda:0", requires_grad=True)
        similarity = similarity - self.thresh # BFP_SHAPE T(2048, 2048)  S matrix
        if not pos_label_mask.any():
            return loss
        # mask out self-contrastive
        log_prob_pos = - self.scale_pos * similarity * pos_label_mask
        log_prob_pos = torch.exp(log_prob_pos).sum(1)   # BFP_SHAPE T(2048, 2048)
        log_prob_pos = torch.log(log_prob_pos + 1) / self.scale_pos   # BFP_SHAPE T(2048, 2048)
        

        keep = ious >= self.iou_threshold
        log_prob_pos = log_prob_pos[keep] # BFP_SHAPE T(34)
        coef = self.reweight_func(ious[keep])
        loss = (log_prob_pos * coef).mean()

        if not neg_label_mask.any():
            return loss
        log_prob_neg =   self.scale_neg * similarity * neg_label_mask
        log_prob_neg = torch.exp(log_prob_neg).sum(1)   # BFP_SHAPE T(2048, 2048)
        log_prob_neg = torch.log(log_prob_neg + 1) / self.scale_neg   # BFP_SHAPE T(2048, 2048)
        log_prob_neg = log_prob_neg.mean()
        return loss + log_prob_neg

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay
if __name__ == "__main__":
    x = torch.ones((12, 12))