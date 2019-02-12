import torch
import torch.nn.functional as F

# logit: B x 1 x H x W
def generate_mask_from_logit(logit):
    prob = F.sigmoid(logit)
    mask = torch.bernoulli(prob)
    return prob, mask

# logit: B x 1 x H x W
# target_loss: B
class mask_generator_loss_from_logit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, mask, target_loss):
        prob = F.sigmoid(logit)
        # print(mask.size())
        # print(target_loss.size())
        ctx.save_for_backward((mask - prob)*target_loss.view(-1,1,1,1))
        return prob.new_zeros(())
    @staticmethod
    def backward(ctx, grad_output):
        g, = ctx.saved_tensors
        return grad_output*g, None, None
