import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")
import math
import os

import optuna
import utilities
import torch.optim as optim
from torch.autograd import Variable


DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample, diff_net, vae_net, diff_latent_dim, noise_divider=1.0, timesteps:str=None, n_timesteps=None):
    # x_T ~ N(0, 1), sample initial noise
    # samples = torch.randn(n_sample, N_ITEMS).to(DEVICE)
    diff_net.eval()
    vae_net.eval()
    with torch.no_grad():

        # Randomly sample timesteps for each sample
        if timesteps == 'random':
            encode_x = torch.randn(n_sample, diff_latent_dim).to(DEVICE)  # sample from prior, pure gaussian noise

            for j in range(n_sample):
                timesteps = np.random.randint(1, n_timesteps)
                for i in range(timesteps, 0, -1):

                    # sample some random noise to inject back in. For i = 1, don't add back in noise
                    z = torch.randn(diff_latent_dim).to(DEVICE) * noise_divider if i > 1 else 0
                    pred_noise_eps = diff_net.forward(torch.unsqueeze(encode_x[j], dim=0), torch.as_tensor([i], device=DEVICE)) # predict noise e_(x_t,t)
                    encode_x[j] = torch.squeeze(denoise_add_noise(torch.unsqueeze(encode_x[j], dim=0), torch.as_tensor([i], device=DEVICE), pred_noise_eps, z))
            samples = vae_net.decode(encode_x)
        else:
            encode_x = torch.randn(n_sample, diff_latent_dim).to(DEVICE)

            for i in range(n_timesteps, 0, -1):

                # sample some random noise to inject back in. For i = 1, don't add back in noise
                z = torch.randn_like(encode_x) * noise_divider if i > 1 else 0

                pred_noise_eps = diff_net.forward(encode_x, torch.full((n_sample,), i, device=DEVICE, dtype=torch.long))  # predict noise e_(x_t,t)
                encode_x = denoise_add_noise(encode_x, i, pred_noise_eps, z)

            samples = vae_net.decode(encode_x)
    return samples


def resume(model, filename, VAE_DIR_PATH):
    # print('Loading model parameters from %s' % filename)
    try:
        model.load_state_dict(torch.load(os.path.normpath(os.path.join(VAE_DIR_PATH, filename))))
    except:
        print('Failed to load model parameters from %s' % filename)
        raise optuna.TrialPruned()


def checkpoint(model, filename, VAE_DIR_PATH):
    """Save model parameters to file"""
    #print('Saving model parameters to %s' % filename)
    try:
        torch.save(model.state_dict(), os.path.normpath(os.path.join(VAE_DIR_PATH, filename)))
    except:
        print('Failed to save model parameters to %s' % filename)
        raise optuna.TrialPruned()


class SDRM(nn.Module):
    def __init__(self, N_ITEMS, EMB_DIM, LATENT_DIM=200, n_hidden_layers=4):
        super(SDRM, self).__init__()
        self.emb_layer = nn.Linear(EMB_DIM, EMB_DIM)
        self.EMB_DIM = EMB_DIM
        self.n_hidden_layers = n_hidden_layers

        self.dnn = nn.Sequential(nn.Linear(N_ITEMS + EMB_DIM, LATENT_DIM), nn.PReLU(),
                      *([nn.Linear(LATENT_DIM, LATENT_DIM), nn.PReLU()] * n_hidden_layers),
                      nn.Linear(LATENT_DIM, N_ITEMS), nn.Tanh())

    def forward(self, x, t):
        time_emb = self.timestep_embedding(t, self.EMB_DIM)
        emb = self.emb_layer(time_emb)
        x = F.dropout(x, p=0.5)
        x = torch.cat([x, emb], dim=-1)
        x = self.dnn(x)
        return x

    def timestep_embedding(self, timesteps, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=DEVICE)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


def train_variational_autoencoder(model, train_data, test_data, epochs, batch_size, lr, early_stop_metric='NDCG@50', VAE_DIR_PATH='./'):
    anneal_cap = 0.2
    anneal_count = 0.0
    best_metric = -np.inf
    best_epoch = 0
    early_stop_counter = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        model.is_training = 1
        train_data = train_data[np.random.permutation(train_data.shape[0])]  # Shuffle data
        for start_idx in range(0, train_data.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, train_data.shape[0])
            anneal = min(anneal_cap, 1. * anneal_count / 20_000)

            X = torch.tensor(train_data[start_idx:end_idx].toarray(), dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            output, vae_kl = model(X)

            #mse_loss = F.mse_loss(output, X)
            log_softmax_var = F.log_softmax(output, dim=1)
            neg_ll = - torch.mean(torch.sum(log_softmax_var * X, dim=1))

            l2_reg = model.get_l2_reg()

            loss = neg_ll + anneal * vae_kl + l2_reg
            loss.backward()
            optimizer.step()
            anneal_count += 1

        #print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
        # Evaluate
        model.eval()
        model.is_training = 0
        eval_metric_list = []
        valid_train, valid_test = utilities.split_train_test_proportion_from_csr_matrix(test_data, batch_size=1000)
        for start_idx in range(0, valid_train.shape[0], 500):
            end_idx = min(start_idx + 500, valid_train.shape[0])
            X = valid_train[start_idx:end_idx]
            X_pred, _ = model(torch.tensor(X.toarray(), dtype=torch.float32, device=DEVICE))
            X_pred = X_pred.detach().cpu().numpy()

            masked_output = utilities.mask_training_examples(sparse_training_set=X, dense_matrix=X_pred)
            if 'Recall' in early_stop_metric:
                eval_metric = utilities.recall_at_k_batch(masked_output, valid_test[start_idx:end_idx],
                                                   k=int(early_stop_metric.split('@')[1]))
            else:
                eval_metric = utilities.NDCG_binary_at_k_batch(masked_output, valid_test[start_idx:end_idx],
                                                   k=int(early_stop_metric.split('@')[1]))
            eval_metric_list.append(eval_metric)
        avg_metric = np.nanmean(np.concatenate(eval_metric_list))
        if avg_metric > best_metric:
            best_metric = max(best_metric, avg_metric)
            checkpoint(model, f"epoch-{epoch}.pth", VAE_DIR_PATH)
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > 20:
                break

    resume(model, f"epoch-{best_epoch}.pth", VAE_DIR_PATH)
    model.model_is_trained = True
    model.is_training = 0


def score_matching_loss(model, XT, t, epsilon_theta, epsilon, mu):
    XT.requires_grad_(True)
    score_x = model(XT, t)
    perturbed_x = XT + mu * epsilon
    perturbed_score_x = model(perturbed_x, t)
    score_diff = (perturbed_score_x - score_x) / (mu ** 2)
    residual = epsilon_theta - XT
    loss = 0.5 * (F.mse_loss(score_diff, residual) + F.mse_loss(residual, score_x)) / (1e-8 + residual.var())
    return loss

# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return torch.as_tensor(ab_t.sqrt()[t, None] * x + (1 - ab_t[t, None]) * noise, dtype=torch.float)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, p_drop=0.5):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, self.latent_dim*2, bias=True))
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, hidden_dim, bias=True),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, input_dim, bias=True))

        self.model_is_trained = False
        self.is_training = 0  # Prevents stochastic behavior in eval() mode

        self.dropout = nn.Dropout(p=p_drop, inplace=False)

        self.weight_decay = 0
        self.cuda2 = True

        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        z, kl_divergence = self.encode(x)
        x = self.decode(z)
        return x, kl_divergence

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=DEVICE, requires_grad=False)
        return mu + self.is_training * eps * std

    def encode(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout(x)
        x = self.encoder(x)
        mu_q, logvar_q = torch.chunk(x, chunks=2, dim=1)

        kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=1))
        #kl_divergence = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), dim=1))
        z = self.reparameterize(mu_q, logvar_q)
        return z, kl_divergence

    def decode(self, z):
        x = self.decoder(z)
        return x

    def get_l2_reg(self):
        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        if self.weight_decay > 0:
            for k, m in self.state_dict().items():
                if k.endswith('.weight'):
                    l2_reg = l2_reg + torch.norm(m, p=2) ** 2
        if self.cuda2:
            l2_reg = l2_reg.cuda()
        return self.weight_decay * l2_reg[0]

    def sample(self, n_samples):
        z = torch.randn(n_samples, self.latent_dim).to(DEVICE)
        return self.decode(z).cpu().detach().numpy()


def train_SDRM(dl, # Dataloader
               N_ITEMS, VAE_HIDDEN, VAE_LATENT, VAE_BATCH_SIZE, VAE_LR, DIFF_LATENT, N_HIDDEN_MLP_LAYERS, DIFF_LR,
               DIFF_TRAINING_EPOCHS, TIMESTEPS, noise_divider, VAE_DIR_PATH, TRAIN_PARTIAL_VALID_DATA,
                VALID_DATA, OPTIMIZATION_OBJECTIVE):
    beta1 = 1e-4
    beta2 = 0.02
    #print("training SDRM")

    # Train the VAE
    variational_ae = VAE(input_dim=N_ITEMS, hidden_dim=VAE_HIDDEN, latent_dim=VAE_LATENT)
    variational_ae.to(device=DEVICE)
    assert variational_ae.model_is_trained is False

    # Train the autoencoder
    train_variational_autoencoder(variational_ae, train_data=TRAIN_PARTIAL_VALID_DATA, test_data=VALID_DATA, epochs=500,
                                  batch_size=VAE_BATCH_SIZE, lr=VAE_LR, early_stop_metric=OPTIMIZATION_OBJECTIVE,
                                  VAE_DIR_PATH=VAE_DIR_PATH)

    assert variational_ae.model_is_trained
    # Freeze weights of vae
    for param in variational_ae.parameters():
        param.requires_grad = False

    variational_ae.eval()  # Set to eval mode to turn off dropout

    # construct DDPM noise schedule
    global b_t
    global a_t
    global ab_t
    b_t = (beta2 - beta1) * torch.linspace(0, 1, TIMESTEPS + 1, device=DEVICE) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    DIFF = SDRM(N_ITEMS=VAE_LATENT, EMB_DIM=TIMESTEPS, LATENT_DIM=DIFF_LATENT, n_hidden_layers=N_HIDDEN_MLP_LAYERS)
    DIFF.to(DEVICE)
    DIFF.train()

    diff_optim = torch.optim.Adam(DIFF.parameters(), lr=DIFF_LR, weight_decay=0.0001, eps=1e-8)  # l2 regularization

    for ep in range(DIFF_TRAINING_EPOCHS):

        # linearly decay learning rate
        diff_optim.param_groups[0]['lr'] = DIFF_LR * (1 - ep / DIFF_TRAINING_EPOCHS)

        # Training
        average_loss = []

        for x, _ in iter(dl):
            diff_optim.zero_grad()
            encode_x, _ = variational_ae.encode(x.to_dense().to(DEVICE))

            # perturb data
            noise = torch.randn_like(encode_x, dtype=torch.float) * noise_divider
            t = torch.randint(1, TIMESTEPS + 1, (encode_x.shape[0],)).to(DEVICE)
            x_pert = perturb_input(encode_x, t, noise)

            # use network to recover noise
            pred_noise = DIFF.forward(x_pert, t)

            diff_loss = score_matching_loss(DIFF, XT=encode_x, t=t, epsilon_theta=pred_noise, epsilon=noise, mu=.1)

            average_loss.append(diff_loss.detach().item())
            diff_loss.backward()
            diff_optim.step()

    return DIFF, variational_ae
