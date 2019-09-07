def criterion_pixel(logit_pixel, truth_pixel):
    batch_size = len(logit_pixel)
    logit = logit_pixel.view(batch_size,-1)
    truth = truth_pixel.view(batch_size,-1)
    assert(logit.shape==truth.shape)
    loss = soft_dice_criterion(logit, truth)
    loss = loss.mean()
    return loss


def soft_dice_criterion(logit, truth, weight=[0.2,0.8]):
    batch_size = len(logit)
    probability = torch.sigmoid(logit)

    p = probability.view(batch_size,-1)
    t = truth.view(batch_size,-1)
    w = truth.detach()
    w = w*(weight[1]-weight[0])+weight[0]

    p = w*(p*2-1)  #convert to [0,1] --&gt; [-1, 1]
    t = w*(t*2-1)

    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  = 1 - 2*intersection/union

    loss = dice
    return loss

def compute_lovasz_gradient(truth): #sorted
    truth_sum    = truth.sum()
    intersection = truth_sum - truth.cumsum(0)
    union        = truth_sum + (1 - truth).cumsum(0)
    jaccard      = 1. - intersection / union
    T = len(truth)
    jaccard[1:T] = jaccard[1:T] - jaccard[0:T-1]

    gradient = jaccard
    return gradient

def lovasz_hinge_one(logit , truth):

    m = truth.detach()
    m = m*(margin[1]-margin[0])+margin[0]

    truth = truth.float()
    sign  = 2. * truth - 1.
    hinge = (m - logit * sign)
    hinge, permutation = torch.sort(hinge, dim=0, descending=True)
    hinge = F.relu(hinge)

    truth = truth[permutation.data]
    gradient = compute_lovasz_gradient(truth)

    loss = torch.dot(hinge, gradient)
    return loss

def lovasz_loss(logit, truth, margin=[1,5]):
    lovasz_one = lovasz_hinge_one
    batch_size = len(truth)
    loss = torch.zeros(batch_size).cuda()
    for b in range(batch_size):
        l, t = logit[b].view(-1), truth[b].view(-1)
        loss[b] = lovasz_one(l, t)
    return loss