import torch


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
    # pred   : [bs, height, width] -> [bs, width * height]
    # target : [bs, height, width] -> [bs, width * height]
    height, width = pred.size(1), pred.size(2)
    pred, target = pred.view(-1, width * height), target.view(-1, width * height)

    # intersection : [bs,]
    # dice_coef    : [bs,]
    intersection = (pred * target).sum(dim=1)
    dice_coef = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice_coef


def intersection_over_union(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
    # pred   : [bs, height, width] -> [bs, width * height]
    # target : [bs, height, width] -> [bs, width * height]
    height, width = pred.size(1), pred.size(2)
    pred, target = pred.view(-1, width * height), target.view(-1, width * height)

    intersection = (pred * target).sum(dim=1)
    union = torch.maximum(pred, target).sum(dim=1)
    iou = (intersection + smooth) / (union + smooth)
    return iou


def hausdorff_distance_2d(pred: torch.Tensor, target: torch.Tensor):
    pred, target = pred.float(), target.float()

    # Calculate the distance matrix
    distance_matrix = torch.cdist(pred, target, p=2)

    # Find the Hausdorff distance
    value1 = distance_matrix.min(dim=2).values.max(dim=1, keepdim=True).values
    value2 = distance_matrix.min(dim=1).values.max(dim=1, keepdim=True).values
    value = torch.cat((value1, value2), dim=1)
    hausdorff_distance = value.max(dim=1).values
    min_distance = distance_matrix.min()
    # Normalize the results
    max_distance = distance_matrix.max()
    normalized_hausdorff_distance = (hausdorff_distance -min_distance)/ (max_distance-min_distance)

    return normalized_hausdorff_distance


if __name__ == '__main__':
    x = torch.zeros((8, 8))
    y = torch.zeros((8, 8))
    x[2:6, 2:6] = 1
    y[3:7, 4:6] = 1
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    print(f"Dice Coefficient: {dice_coefficient(x, y)}")
    print(f"Intersection over Union: {intersection_over_union(x, y)}")
    print(f"2D Hausdorff Distance: {hausdorff_distance_2d(x, y)}")
