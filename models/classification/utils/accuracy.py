import torch


def accuracy(output, target, topk=(1,)):
    '''Computes the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)

        # Total element
        batch_size = target.size(0)

        # The  topk function selects the top k number of output
        # values means element
        # pred means index
        values, pred = output.topk(maxk, 1, True, True)

        # Transpose
        pred = pred.t()

        # correct: tensor([[True, True, False, False],
        #                  [False, False, True, True],
        #                  [False, False, False, False]])
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k means the total number of elements meeting requirements
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # [] %
            res.append(correct_k.mul(100.0 / batch_size))
        return res


if __name__ == '__main__':
    outputs = torch.tensor([[-0.5816, -0.3873, -1.0215, -1.0145, 0.4053],
                            [0.7265, 1.4164, 1.3443, 1.2035, 1.8823],
                            [-0.4451, 0.1673, 1.2590, -2.0757, 1.7255],
                            [0.2021, 0.3041, 0.1383, 0.3849, -1.6311]])
    target = torch.tensor([[4], [4], [2], [1]])
    res = accuracy(outputs, target, topk=(1, 3))
    print("res: ", res)
