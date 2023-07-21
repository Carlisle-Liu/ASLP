import torch

def stochastic_label_perturbation(gt, alpha, beta, noise=False):
    new_label = torch.zeros(gt.shape).cuda()
    prob = torch.rand(gt.shape[:1]).cuda()
    new_label[prob >= alpha] = gt[prob >= alpha]
    new_label[prob < alpha] = (1 - 2 * beta) * gt[prob < alpha] + beta
    if noise == True:
        e = torch.empty_like(gt[prob < alpha])
        torch.nn.init.trunc_normal_(e, mean=0, std=1, a=-0.25, b=0.25)
        new_label[prob < alpha] = new_label[prob < alpha] + e
    return new_label

def stochastic_label_smoothing(gt, beta):
    b = torch.ones_like(gt) * beta
    new_label = (1 - 2 * b) * gt + b
    return new_label



