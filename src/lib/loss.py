import torch


class ContrastiveLoss(torch.nn.Module):
    def __int__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, score, label, margin):
        loss = torch.mean((1 - label) * 0.5 * torch.pow(score, 2) +
                          label * 0.5 * torch.pow(torch.clamp(margin - score, min=0.0), 2))
        return loss


class TripletLoss(torch.nn.Module):
    def __int__(self):
        super(TripletLoss, self).__init__()

    def forward(self, score_positive, score_negative, margin):
        loss = torch.mean(torch.clamp(torch.pow(score_positive, 2) - torch.pow(score_negative, 2) + margin, min=0.0))
        return loss