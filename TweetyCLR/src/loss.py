'''
InfoNCE Loss Function. Borrowed from Mathis' CEBRA paper
'''
import torch
import torch.nn.functional as F

from torch import einsum, logsumexp, no_grad

def info_nce(ref, pos, neg, tau = 1.0):
    pos_dist = einsum("nd,nd->n", ref, pos)/tau 
    neg_dist = einsum("nd,md->nm", ref, neg)/tau 
    with no_grad():
        c, _ = neg_dist.max(dim=1)
    pos_dist = pos_dist - c.detach()
    neg_dist = neg_dist - c.detach()
    pos_loss = -pos_dist.mean()
    neg_loss = logsumexp(neg_dist, dim = 1).mean()
    return pos_loss + neg_loss

# def info_nce(ref, pos, neg, tau=1.0):
#     """
#     InfoNCE loss function.
    
#     Args:
#     - ref (torch.Tensor): Embeddings of anchor samples of shape (batch_size, embedding_dim).
#     - pos (torch.Tensor): Embeddings of positive samples of shape (batch_size, embedding_dim).
#     - neg (torch.Tensor): Embeddings of negative samples of shape (batch_size, embedding_dim).
#     - tau (float): Temperature parameter for scaling logits (default=1.0).
    
#     Returns:
#     - loss (torch.Tensor): The InfoNCE loss.
#     """
#     # Concatenate anchor, positive, and negative embeddings
#     embeddings = torch.cat([ref, pos, neg], dim=0)
    
#     # Calculate pairwise cosine similarities
#     similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    
#     # Calculate logits for positive pairs (anchor, positive)
#     positive_logits = torch.diag(similarity_matrix[len(ref):2*len(ref)])
    
#     # Calculate logits for negative pairs (anchor, negative)
#     negative_logits = torch.sum(torch.exp(similarity_matrix[2*len(ref):]), dim=1)
    
#     # Combine logits for positive and negative pairs
#     logits = torch.cat([positive_logits, negative_logits], dim=0)
    
#     # Apply temperature scaling
#     logits /= tau
    
#     # Calculate cross-entropy loss
#     loss = -torch.mean(torch.log(torch.exp(positive_logits) / torch.sum(torch.exp(logits), dim=0)))
    
#     return loss