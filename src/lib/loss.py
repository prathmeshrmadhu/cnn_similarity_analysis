import torch
import numpy as np
import torch.nn.functional as F


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


class CustomLoss(torch.nn.Module):
    def __int__(self):
        super(CustomLoss, self).__init__()

    def forward(self, q1, q2, q3, q4, p1, p2, p3, p4, n1, n2, n3, n4, margin, lam, cos=True):
        if cos:
            score_positive = 1 - F.cosine_similarity(q3, p3)
            score_negative = 1 - F.cosine_similarity(q3, n3)
        else:
            score_positive = F.pairwise_distance(q3, p3, p=2.0)
            score_negative = F.pairwise_distance(q3, n3, p=2.0)
        triplet_loss = torch.mean(
            torch.clamp(torch.pow(score_positive, 2) - torch.pow(score_negative, 2) + margin, min=0.0))
        regular = lam * (F.l1_loss(q1, p1) + F.l1_loss(p2, q2) + F.l1_loss(q4, p4))
        loss = triplet_loss + regular
        return loss


class CustomLoss_vgg(torch.nn.Module):
    def __int__(self):
        super(CustomLoss_vgg, self).__init__()

    def forward(self, q1, q2, q3, q4, q5, q6, p1, p2, p3, p4, p5, p6, n6, margin, lam, cos=True):
        if cos:
            score_positive = 1 - F.cosine_similarity(q6, p6)
            score_negative = 1 - F.cosine_similarity(q6, n6)
        else:
            score_positive = F.pairwise_distance(q6, p6, p=2.0)
            score_negative = F.pairwise_distance(q6, n6, p=2.0)
        triplet_loss = torch.mean(
            torch.clamp(torch.pow(score_positive, 2) - torch.pow(score_negative, 2) + margin, min=0.0))
        d1 = F.pairwise_distance(q1, p1, p=2.0)
        d2 = F.pairwise_distance(q2, p2, p=2.0)
        d3 = F.pairwise_distance(q3, p3, p=2.0)
        d4 = F.pairwise_distance(q4, p4, p=2.0)
        d5 = F.pairwise_distance(q5, p5, p=2.0)
        regular = lam * (d1 + d2 + d3 + d4 + d5)
        # regular = lam * (F.l1_loss(q1, p1) + F.l1_loss(p2, q2) + F.l1_loss(q3, p3) + F.l1_loss(q4, p4) + F.l1_loss(q5, p5))
        loss = triplet_loss + regular
        return loss


class SimCLR_Loss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(SimCLR_Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(dim=0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        # SIMCLR
        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()  # .float()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss