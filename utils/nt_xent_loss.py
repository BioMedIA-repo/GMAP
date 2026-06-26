import numpy as np
import torch


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, temperature=0.5, use_cosine_similarity=True, beta=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.beta = beta

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            return self._cosine_similarity
        else:
            return self._dot_similarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_similarity(x, y):
        return torch.matmul(x, y.T)

    @staticmethod
    def _cosine_similarity(x, y):
        return torch.nn.functional.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        batch_size = zis.shape[0]
        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)

        # Filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    feature_dim = 128

    # Generate random zis and zjs tensors
    zis = torch.randn(batch_size, feature_dim).to(device)
    zjs = torch.randn(batch_size, feature_dim).to(device)

    # Instantiate the loss function
    loss_fn = NTXentLoss(device=device, temperature=0.5, use_cosine_similarity=True)

    # Calculate the loss
    loss = loss_fn(zis, zjs)
    print(f"Final loss: {loss.item()}")

if __name__ == "__main__":
    main()
