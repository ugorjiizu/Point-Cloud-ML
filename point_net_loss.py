'''
Our dataset is heavily imbalanced, with common categories like "Ground" and "Building" significantly outnumbering the other classes. If we were to train using a standard loss function like Cross Entropy, it would treat all classes as equally important. However, in practice, we want our model to pay more attention to the less common categories and not overly prioritize the more prevalent ones.

To address this imbalance, we will use a custom weighted loss function, adapted from the repository we referenced for the PointNet model [source](https://github.com/itberrios/3D/blob/main/point_net/point_net_loss.py). This approach helps the model focus on the more challenging classes (hard examples) while giving less emphasis to the easier ones.

The core of our loss function is based on Focal Loss, an enhancement of Cross Entropy that is particularly effective for imbalanced datasets. Focal Loss shifts the focus to the sparse set of hard examples, ensuring that the model learns more effectively from them.

In addition, we will incorporate Dice Loss to further improve the Intersection over Union (IoU) performance, ensuring better segmentation accuracy across all classes still following the repository's code.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# special loss for segmentation Focal Loss + Dice Loss
class PointNetSegLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, size_average=True, dice=False):
        super(PointNetSegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.dice = dice

        # sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # get Balanced Cross Entropy Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)
        

    def forward(self, predictions, targets, pred_choice=None):

        # get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions.transpose(2, 1), targets)

        # reformat predictions (b, n, c) -> (b*n, c)
        predictions = predictions.contiguous() \
                                 .view(-1, predictions.size(2)) 
        # get predicted class probabilities for the true class
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average: loss = loss.mean() 
        else: loss = loss.sum()

        # add dice coefficient if necessary
        if self.dice: return loss + self.dice_loss(targets, pred_choice, eps=1)
        else: return loss


    @staticmethod
    def dice_loss(predictions, targets, eps=1):
        ''' Compute Dice loss, directly compare predictions with truth '''

        targets = targets.reshape(-1)
        predictions = predictions.reshape(-1)

        cats = torch.unique(targets)

        top = 0
        bot = 0
        for c in cats:
            locs = targets == c

            # get truth and predictions for each class
            y_tru = targets[locs]
            y_hat = predictions[locs]

            top += torch.sum(y_hat == y_tru)
            bot += len(y_tru) + len(y_hat)


        return 1 - 2*((top + eps)/(bot + eps)) 