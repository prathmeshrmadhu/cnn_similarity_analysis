import torch
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

    def forward(self, q1, q2, q3, q4, p1, p2, p3, p4, n1, n2, n3, n4, margin, lam):
        score_positive = 1 - F.cosine_similarity(q3, p3)
        score_negative = 1 - F.cosine_similarity(q3, n3)
        triplet_loss = torch.mean(
            torch.clamp(torch.pow(score_positive, 2) - torch.pow(score_negative, 2) + margin, min=0.0))
        regular = lam * (F.l1_loss(q1, p1) + F.l1_loss(p2, q2) + F.l1_loss(q4, p4))
        loss = triplet_loss + regular
        return loss


class ContrastiveLossSimClr(torch.nn.Module):
    def __int__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze_(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss