import torch
import matplotlib.pyplot as plt


class SimFnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 5)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


class FRHistory:
    def __init__(self, params):
        self.params = list(params)
        self.grads = [torch.zeros_like(p) for p in self.params]
        self.old_grads = None

    def step(self, grads):
        self.old_grads = self.grads
        self.grads = list(grads)
        return self

    def get_direction(self):
        if self.old_grads is None:
            return [-grad for grad in self.grads]

        beta = sum(torch.sum(grad * (grad - old_grad)) for grad, old_grad in zip(self.grads, self.old_grads))
        beta /= sum(torch.sum(old_grad ** 2) for old_grad in self.old_grads)
        return [-grad + beta * old_grad for grad, old_grad in zip(self.grads, self.old_grads)]


x = torch.arange(0, 4.05, 0.05)
xlen = len(x)
y = torch.zeros(xlen)

for n in range(xlen):
    xs = x[n]
    y[n] = xs + torch.sqrt(xs)

P = x.unsqueeze(1)
T = y.unsqueeze(1)

model = SimFnModel()
history = FRHistory(model.parameters())
criterion = torch.nn.MSELoss()

losses = []
for epoch in range(10):
    optimizer_direction = history.get_direction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0, nesterov=False)
    optimizer.param_groups[0]['params'] = model.parameters()
    optimizer_state = {'momentum_buffer': optimizer_direction}

    optimizer.zero_grad()
    outputs = model(P)
    loss = criterion(outputs, T)

    loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    history.step(grads)
    optimizer_direction = history.get_direction()
    optimizer.param_groups[0]['params'] = model.parameters()
    optimizer_state['momentum_buffer'] = optimizer_direction
    optimizer.step()
    losses.append(loss.item())
plt.plot(range(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(P.numpy(), T.numpy(), 'x', label='Real function')
plt.plot(P.numpy(), model(P).detach().numpy(), label='Expected approximation')
plt.plot(P.numpy(), (model(P) - T).detach().numpy(), label='Loss')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Sim Fn')
plt.legend()
plt.grid()
plt.show()
