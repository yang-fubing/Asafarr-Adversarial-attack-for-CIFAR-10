import torch


# perform fgsm attack
def fgsm(model, x, y, loss_fn, epsilon):
    x_adv = x.detach().clone()
    x_adv.requires_grad = True
    loss = loss_fn(model(x_adv), y)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.detach().sign()
    return x_adv


# perform ifgsm attack
def ifgsm(model, x, y, loss_fn, epsilon, alpha, num_iter=30):
    x_adv = x.detach().clone()
    for i in range(num_iter):
        x_adv = fgsm(model, x_adv, y, loss_fn, alpha)
        x_adv = torch.min(torch.max(x_adv, x-epsilon), x+epsilon)
    return x_adv


# perform mifgsm attack
def mifgsm(model, x, y, loss_fn, epsilon, alpha, num_iter=30, mu=0.9):
    x_adv = x.detach().clone()
    g = torch.zeros_like(x_adv, device=x.device)
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        x_adv_norm = torch.norm(x_adv, p=1)
        g = mu*g + x_adv.grad.detach()/x_adv_norm
        x_adv = x_adv + alpha * g.detach().sign()
        x_adv = torch.min(torch.max(x_adv, x-epsilon), x+epsilon)
    return x_adv
