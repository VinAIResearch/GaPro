import gpytorch
import torch
import torch.nn as nn
import torch_scatter
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch.utils.data import DataLoader, TensorDataset


class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        # variational_strategy = UnwhitenedVariationalStrategy(self, train_x, variational_distribution, learn_inducing_locations=False)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


def fit_gp(
    coords_float, feats, spp, b1_inds, b2_inds, intersect_inds, training_iter=50, npoint_nearest=800, spp_pool=True
):

    # try:
    # b1_feats, b2_feat, intersect_feat, b1_spp, b2_spp, intersect_spp, training_iter=50):
    device = feats.device

    intersect_coords = coords_float[intersect_inds]
    intersect_feats = feats[intersect_inds]
    # intersect_spp = spp[intersect_inds]
    intersect_centroid = intersect_coords.mean(0)

    b1_coords = coords_float[b1_inds]
    b1_feats = feats[b1_inds]
    b1_spp = spp[b1_inds]

    b2_coords = coords_float[b2_inds]
    b2_feats = feats[b2_inds]
    b2_spp = spp[b2_inds]

    if not spp_pool:
        if len(b1_inds) > npoint_nearest:
            dist_to_centroid = ((b1_coords - intersect_centroid[None, :]) ** 2).sum(1)
            topk_inds = torch.topk(dist_to_centroid, k=npoint_nearest, largest=False)[1]

            b1_feats = b1_feats[topk_inds]
            b1_spp = b1_spp[topk_inds]

        if len(b2_inds) > npoint_nearest:
            dist_to_centroid = ((b2_coords - intersect_centroid[None, :]) ** 2).sum(1)
            topk_inds = torch.topk(dist_to_centroid, k=npoint_nearest, largest=False)[1]

            b2_feats = b2_feats[topk_inds]
            b2_spp = b2_spp[topk_inds]

    if spp_pool:
        _, b1_spp = torch.unique(b1_spp, return_inverse=True)
        b1_feats = torch_scatter.scatter(b1_feats, b1_spp[:, None].expand(-1, b1_feats.shape[1]), dim=0, reduce="mean")

        _, b2_spp = torch.unique(b2_spp, return_inverse=True)
        b2_feats = torch_scatter.scatter(b2_feats, b2_spp[:, None].expand(-1, b2_feats.shape[1]), dim=0, reduce="mean")

        # _, intersect_spp = torch.unique(intersect_spp, return_inverse=True)
        # intersect_feats = torch_scatter.scatter(intersect_feats, intersect_spp[:, None].expand(-1, intersect_feats.shape[1]), dim=0, reduce="mean")

    # scaler = torch.cuda.amp.GradScaler(enabled=True)

    train_x = torch.cat([b1_feats, b2_feats], dim=0)
    train_y = torch.cat(
        [-1 * torch.ones(b1_feats.shape[0], device=device), torch.ones(b2_feats.shape[0], device=device)], dim=0
    )

    # print(train_x.shape)
    model = GPClassificationModel(train_x).to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the amount of training data
    mll = VariationalELBO(likelihood, model, train_y.numel())

    for i in range(training_iter):
        output = model(train_x)
        loss = -mll(output, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Go into eval mode
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        observed_pred = likelihood(model(intersect_feats))
        pred_probs = observed_pred.mean
        pred_labels = pred_probs.ge(0.5)
        pred_variance = observed_pred.variance

        pred_probs_new = torch.where(pred_labels == 1, pred_probs, 1 - pred_probs)

    return pred_probs, pred_probs_new, pred_labels, pred_variance


def fit_gp_ensemble(
    coords_float,
    feats,
    spp,
    b1_inds,
    b2_inds,
    intersect_inds,
    channel_dims,
    training_iter=50,
    npoint_nearest=800,
    spp_pool=True,
):
    device = feats.device

    intersect_coords = coords_float[intersect_inds]
    intersect_feats = feats[intersect_inds]
    intersect_spp = spp[intersect_inds]
    intersect_centroid = intersect_coords.mean(0)

    b1_coords = coords_float[b1_inds]
    b1_feats = feats[b1_inds]
    b1_spp = spp[b1_inds]

    b2_coords = coords_float[b2_inds]
    b2_feats = feats[b2_inds]
    b2_spp = spp[b2_inds]

    if len(b1_inds) > npoint_nearest:
        dist_to_centroid = ((b1_coords - intersect_centroid[None, :]) ** 2).sum(1)
        topk_inds = torch.topk(dist_to_centroid, k=npoint_nearest, largest=False)[1]

        b1_feats = b1_feats[topk_inds]
        b1_spp = b1_spp[topk_inds]

    if len(b2_inds) > npoint_nearest:
        dist_to_centroid = ((b2_coords - intersect_centroid[None, :]) ** 2).sum(1)
        topk_inds = torch.topk(dist_to_centroid, k=npoint_nearest, largest=False)[1]

        b2_feats = b2_feats[topk_inds]
        b2_spp = b2_spp[topk_inds]

    if spp_pool:
        _, b1_spp = torch.unique(b1_spp, return_inverse=True)
        b1_feats = torch_scatter.scatter(b1_feats, b1_spp[:, None].expand(-1, b1_feats.shape[1]), dim=0, reduce="mean")

        _, b2_spp = torch.unique(b2_spp, return_inverse=True)
        b2_feats = torch_scatter.scatter(b2_feats, b2_spp[:, None].expand(-1, b2_feats.shape[1]), dim=0, reduce="mean")

        _, intersect_spp = torch.unique(intersect_spp, return_inverse=True)
        intersect_feats = torch_scatter.scatter(
            intersect_feats, intersect_spp[:, None].expand(-1, intersect_feats.shape[1]), dim=0, reduce="mean"
        )

    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    # print(intersect_feats.shape, b1_feats.shape)
    pred_probs_2labels = torch.zeros((intersect_feats.shape[0], 2), dtype=torch.float, device=device)
    pred_variance = torch.zeros((intersect_feats.shape[0]), dtype=torch.float, device=device)

    for i in range(len(channel_dims) - 1):
        c_start, c_end = channel_dims[i], channel_dims[i + 1]

        b1_feats_ = b1_feats[:, c_start:c_end]
        b2_feats_ = b2_feats[:, c_start:c_end]
        intersect_feats_ = intersect_feats[:, c_start:c_end]

        train_x = torch.cat([b1_feats_, b2_feats_], dim=0)
        train_y = torch.cat(
            [-1 * torch.ones(b1_feats_.shape[0], device=device), torch.ones(b2_feats_.shape[0], device=device)], dim=0
        )

        model = GPClassificationModel(train_x).to(device)
        likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the amount of training data
        mll = VariationalELBO(likelihood, model, train_y.numel())

        for i in range(training_iter):
            # Zero backpropped gradients from previous iteration
            # optimizer.zero_grad()
            # Get predictive output
            output = model(train_x)
            loss = -mll(output, train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calc loss and backprop gradients
            # loss.backward()

            # if i % 5 == 0:
            #     print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))

            # optimizer.step()

        # Go into eval mode
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            observed_pred_ = likelihood(model(intersect_feats_))
            pred_probs_ = observed_pred_.mean
            pred_labels_ = pred_probs_.ge(0.5)
            pred_variance_ = observed_pred_.variance

            # pred_pos_probs_ = pred_probs_
            # pred_neg_probs_ = 1 - pred_probs_

            # pred_probs_2labels[pred_labels_ == 0, 0]
            # print(pred_probs_2labels.shape, pred_labels_.shape, intersect_feats_.shape)
            pred_probs_2labels[:, 1] += torch.where(pred_labels_ == 1, pred_probs_, 1 - pred_probs_)
            pred_probs_2labels[:, 0] += torch.where(pred_labels_ == 1, 1 - pred_probs_, pred_probs_)

            pred_variance += pred_variance_

    pred_probs, pred_labels = torch.max(pred_probs_2labels, dim=1)

    # pred_labels = torch.randn((intersect_feats.shape[0])) >= 0.5
    # pred_probs = torch.randn((intersect_feats.shape[0])) >= 0.5
    if spp_pool:
        pred_probs = pred_probs[intersect_spp]
        pred_labels = pred_labels[intersect_spp]
        pred_variance = pred_variance[intersect_spp]

    return pred_probs, pred_labels, pred_variance


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        n_hidden1 = 128
        n_hidden2 = 128
        self.layer_1 = nn.Linear(32, n_hidden1)
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x).squeeze(-1)

        return x


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer_1 = nn.Linear(32, 1)

    def forward(self, inputs):
        x = self.layer_1(inputs)
        return x


# def fit_simple_model(train_x, train_y, test_x, device, training_iter=50):
def fit_regression_model(feats, spp, b1_inds, b2_inds, intersect_inds, training_iter=50):
    device = feats.device

    # intersect_coords = coords_float[intersect_inds]
    intersect_feats = feats[intersect_inds]
    # intersect_spp = spp[intersect_inds]
    # intersect_centroid = intersect_coords.mean(0)

    # b1_coords = coords_float[b1_inds]
    b1_feats = feats[b1_inds]
    # b1_spp = spp[b1_inds]

    # b2_coords = coords_float[b2_inds]
    b2_feats = feats[b2_inds]
    # b2_spp = spp[b2_inds]

    # _, b1_spp = torch.unique(b1_spp, return_inverse=True)
    # b1_feats = torch_scatter.scatter(b1_feats, b1_spp[:, None].expand(-1, b1_feats.shape[1]), dim=0, reduce="mean")

    # _, b2_spp = torch.unique(b2_spp, return_inverse=True)
    # b2_feats = torch_scatter.scatter(b2_feats, b2_spp[:, None].expand(-1, b2_feats.shape[1]), dim=0, reduce="mean")

    # _, intersect_spp = torch.unique(intersect_spp, return_inverse=True)
    # intersect_feats = torch_scatter.scatter(intersect_feats, intersect_spp[:, None].expand(-1, intersect_feats.shape[1]), dim=0, reduce="mean")

    train_x = torch.cat([b1_feats, b2_feats], dim=0)
    train_y = torch.cat(
        [torch.zeros(b1_feats.shape[0], device=device), torch.ones(b2_feats.shape[0], device=device)], dim=0
    ).float()

    # train_y = torch.where(train_y == -1, 0, 1).float()
    # Define a batch size ,
    # bs = min(32, train_x.shape[0])
    bs = min(256, train_x.shape[0])
    # Both x_train and y_train can be combined in a single TensorDataset, which will be easier to iterate over and slice
    # y_tensor = y_tensor.unsqueeze(1)
    train_ds = TensorDataset(train_x, train_y)
    # Pytorch’s DataLoader is responsible for managing batches.
    # You can create a DataLoader from any Dataset. DataLoader makes it easier to iterate over batches
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)

    model = RegressionModel().to(device)
    model.train()

    # Use the adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    loss_func = torch.nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(training_iter):
        # Within each epoch run the subsets of data = batch sizes.
        loss_arr = []
        for xb, yb in train_dl:

            # breakpoint()
            with torch.cuda.amp.autocast(enabled=True):
                y_pred = model(xb).squeeze(-1)  # Forward Propagation
                loss = loss_func(y_pred, yb)  # Loss Computation

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # optimizer.zero_grad()         # Clearing all previous gradients, setting to zero
            # loss.backward()               # Back Propagation
            # optimizer.step()              # Updating the parameters
            loss_arr.append(loss.detach())

        mean_loss = torch.mean(torch.tensor(loss_arr))
        # print("Loss in iteration :"+str(epoch)+" is: "+str(mean_loss.item()))
        if mean_loss.item() < 1e-3:
            break

    # Go into eval mode
    model.eval()

    with torch.no_grad():
        pred_probs = model(intersect_feats).sigmoid().squeeze(-1)
        pred_labels = pred_probs.ge(0.5)

        pred_probs_new = torch.where(pred_labels == 1, pred_probs, 1 - pred_probs)

    # pred_probs = pred_probs[intersect_spp]
    # pred_labels = pred_labels[intersect_spp]
    # pred_probs_new = pred_probs_new[intersect_spp]

    return pred_probs, pred_probs_new, pred_labels


def fit_gp_spp(coords_float_spp, feats_spp, b1_inds, b2_inds, intersect_inds, training_iter=50):
    device = feats_spp.device

    # intersect_coords = coords_float_spp[intersect_inds]
    intersect_feats = feats_spp[intersect_inds]
    # intersect_centroid = intersect_coords.mean(0)

    # b1_coords = coords_float_spp[b1_inds]
    b1_feats = feats_spp[b1_inds]

    # b2_coords = coords_float_spp[b2_inds]
    b2_feats = feats_spp[b2_inds]

    train_x = torch.cat([b1_feats, b2_feats], dim=0)
    train_y = torch.cat(
        [-1 * torch.ones(b1_feats.shape[0], device=device), torch.ones(b2_feats.shape[0], device=device)], dim=0
    )
    # train_y = torch.cat([torch.zeros(b1_feats.shape[0], device=device), torch.ones(b2_feats.shape[0], device=device)], dim=0)

    # print(train_x.shape)
    model = GPClassificationModel(train_x).to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the amount of training data
    mll = VariationalELBO(likelihood, model, train_y.numel())

    for i in range(training_iter):
        output = model(train_x)
        loss = -mll(output, train_y)

        # print(i, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Go into eval mode
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        f_pred = model(intersect_feats)
        observed_pred = likelihood(f_pred)
        pred_probs = observed_pred.mean
        pred_labels = pred_probs.ge(0.5)

        pred_mu = f_pred.mean
        pred_variance = f_pred.variance

        pred_probs_new = torch.where(pred_labels == 1, pred_probs, 1 - pred_probs)

        # if spp_pool:
        #     pred_probs = pred_probs[intersect_spp]
        #     pred_labels = pred_labels[intersect_spp]
        #     pred_variance = pred_variance[intersect_spp]

    return pred_probs, pred_probs_new, pred_labels, pred_mu, pred_variance
