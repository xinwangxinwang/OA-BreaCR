import torch
import torch.nn as nn
import torch.nn.functional as F


class risk_BCE_loss(nn.Module):
    """
    Defines for computing the risk prediction loss for time-to-event data.

    This class calculates a custom loss function based on binary
    cross-entropy. It utilizes masking to handle censored data during follow-up periods and allows
    for weighted contributions to the loss computation to reflect varying levels of significance
    across events and time periods. The class is parameterized for flexibility in accounting for
    different datasets and settings.
    """
    def __init__(self, weight_loss=2, batch_size=1, num_pred_years=16):
        super(risk_BCE_loss, self).__init__()
        self.weight_loss = weight_loss
        self.batch_size = batch_size
        self.num_pred_years = num_pred_years
        self.max_followup = num_pred_years

    def forward(self, pred, risk_label, years_last_followup, weights=None):
        """
        Computes the risk prediction loss for time-to-event data.

        This method computes the binary cross-entropy loss considering the predicted probabilities,
        the true risk labels, the years since last follow-up, and optional weights. It applies masking
        depending on the follow-up status for each sample to ensure only valid data points contribute
        to the loss calculation. The loss is normalized by the sum of valid risk masks.

        Parameters:
            pred: torch.Tensor
                The predicted probabilities for each year. This tensor has shape (batch_size, num_pred_years).

            risk_label: torch.Tensor
                The true risk labels indicating the year of event occurrence. This is a 1D tensor with
                the length of batch_size.

            years_last_followup: torch.Tensor
                The last observed follow-up period for each sample. This tensor has length equals to
                batch_size.

            weights: Optional[torch.Tensor]
                An optional tensor representing weights for each year. Its dimensions must be
                compatible with risk_mask when broadcasted.

        Returns:
            torch.Tensor
                The computed binary cross-entropy loss adjusted for the follow-up risk trend and
                weighted by provided criteria if applicable.
        """

        pred = F.softmax(pred, dim=1)
        batch_size, num_pred_years = pred.shape
        followup = num_pred_years - 1
        risk_label = risk_label.cpu().detach().numpy()
        years_last_followup = years_last_followup.cpu().detach().numpy()
        risk_mask = torch.ones((batch_size, self.num_pred_years))
        y_seq = torch.zeros((batch_size, self.num_pred_years))

        risk_label[risk_label > (num_pred_years - 1)] = num_pred_years - 1
        for i in range(batch_size):
            y_seq[i, risk_label[i]] = 1
            # ra = risk_label[i]
            # fa = years_last_followup[i]
            if risk_label[i] == followup and years_last_followup[i] < followup:
                risk_mask[i, years_last_followup[i]+1:] = 0

        if torch.sum(risk_mask.float()) == 0:
            print('wrong!!!!!!!!!',  torch.sum(risk_mask.float()))

        # weights = None
        if weights is not None:
            weights_ = torch.tensor(weights, dtype=torch.float).view(1, -1)
            risk_mask = risk_mask * weights_
            # risk_mask_ = risk_mask * weights_

        loss = F.binary_cross_entropy(
            pred, y_seq.float().cuda(),
            weight=risk_mask.float().cuda(),
            reduction='sum'
        ) / torch.sum(risk_mask.float()) * self.weight_loss

        # loss = F.binary_cross_entropy(
        #     pred, y_seq.float().cuda(),
        #     weight=risk_mask_.float().cuda(),
        #     reduction='sum'
        # ) / torch.sum(risk_mask.float()) * self.weight_loss

        return loss
