import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from model import VariationalAutoEncoder

import os
import glob
import argparse
import statistics
from tqdm import tqdm

DATA_MAX_INTENSITY = 667
DATA_MIN_INTENSITY = 0

def get_model_name(p):
    file_names = glob.glob(os.path.join(p, "*.pth"))
    return file_names[-1]

def pre(x):
    return (x - DATA_MIN_INTENSITY)/(DATA_MAX_INTENSITY - DATA_MIN_INTENSITY)

def post(x):
    return x * (DATA_MAX_INTENSITY - DATA_MIN_INTENSITY) + DATA_MIN_INTENSITY


def load_data(p, batch_size, patch_size):
    np_train = np.load(os.path.join(p, "train.npy")).reshape(-1, patch_size*patch_size*patch_size)
    np_valid = np.load(os.path.join(p, "valid.npy")).reshape(-1, patch_size*patch_size*patch_size)
    np_test  = np.load(os.path.join(p,  "test.npy")).reshape(-1, patch_size*patch_size*patch_size)

    np_train = pre(np_train)
    np_valid = pre(np_valid)
    np_test = pre(np_test)

    torch_train = torch.tensor(np_train, dtype=torch.float)
    torch_valid = torch.tensor(np_valid, dtype=torch.float)
    torch_test  = torch.tensor(np_test,  dtype=torch.float)

    train_loader  = torch.utils.data.DataLoader(torch_train, batch_size=batch_size, shuffle=True)
    valid_loader  = torch.utils.data.DataLoader(torch_valid, batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(torch_test, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def loss(x, x_hat, mean, var, mse):

    KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
    reconstruction = mse(x, x_hat)
    elbo = KL + reconstruction

    return KL, reconstruction, elbo

def step_train(train_dataset, valid_dataset, device, epochs, z_dim, n_hidden, alpha, patch_size, save_model_path):

    writer = SummaryWriter()

    img_size = patch_size * patch_size * patch_size

    vae = VariationalAutoEncoder(
        channel=img_size,
        n_hidden=n_hidden,
        z_dim=z_dim,
    ).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=alpha)

    mse_loss = torch.nn.MSELoss(size_average=False)

    best_loss = 99999

    result = {}
    result["train/KL"] = []
    result["train/MSE"] = []
    result["train/ELBO"] = []
    result["valid/KL"] = []
    result["valid/MSE"] = []
    result["valid/ELBO"] = []

    for epoch in range(epochs):
        print("train step: epoch {}".format(epoch))

        train_KL, train_mse, train_elbo = [], [], []
        for input_img in tqdm(train_dataset):
            input_img = input_img.to(device)

            mu, sig, _, x_hat = vae(input_img)

            KLdiv, mse, ELBO = loss(input_img, x_hat, mu, sig, mse_loss)
            
            train_KL.append(KLdiv.item())
            train_mse.append(mse.mean().item())
            train_elbo.append(ELBO.item())

            vae.zero_grad()
            ELBO.backward()
            optimizer.step()

        result["train/KL"].append(statistics.mean(train_KL))
        result["train/MSE"].append(statistics.mean(train_mse))
        result["train/ELBO"].append(statistics.mean(train_elbo))

        writer.add_scalar("train/KL", result["train/KL"][-1], epoch)
        writer.add_scalar("train/MSE", result["train/MSE"][-1], epoch)
        writer.add_scalar("train/ELBO", result["train/ELBO"][-1], epoch)

        if epoch % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                print("valid step: epoch {}".format(epoch))
                vae.eval()

                valid_KL, valid_mse, valid_elbo = [], [], []
                for input_img in tqdm(valid_dataset):
                    input_img = input_img.to(device)

                    mu, sig, _, x_hat = vae(input_img)
                    KLdiv, mse, ELBO = loss(input_img, x_hat, mu, sig, mse_loss)

                    valid_KL.append(KLdiv.item())
                    valid_mse.append(mse.mean().item())
                    valid_elbo.append(ELBO.item())

                result["valid/KL"].append(statistics.mean(valid_KL))
                result["valid/MSE"].append(statistics.mean(valid_mse))
                result["valid/ELBO"].append(statistics.mean(valid_elbo))

                writer.add_scalar("valid/KL", result["valid/KL"][-1], epoch + 1)
                writer.add_scalar("valid/MSE", result["valid/MSE"][-1], epoch + 1)
                writer.add_scalar("valid/ELBO", result["valid/ELBO"][-1], epoch + 1)

                input_img = input_img.view(-1, 1, args.patch_size, args.patch_size, args.patch_size)
                utils.save_image(
                    input_img.cpu().data[:, :, 4, :, :],
                    os.path.join(args.save_sample_path, f'input_{str(epoch + 1).zfill(5)}.png'),
                    normalize=False,
                    nrow=10,
                    range=(0.0, 1.0),
                )

                x_hat = x_hat.view(-1, 1, args.patch_size, args.patch_size, args.patch_size)
                utils.save_image(
                    x_hat.cpu().data[:, :, 4, :, :],
                    os.path.join(args.save_sample_path, f'output_{str(epoch + 1).zfill(5)}.png'),
                    normalize=False,
                    nrow=10,
                    range=(0.0, 1.0),
                )

                if best_loss > result["valid/ELBO"][-1]:
                    best_loss = result["valid/ELBO"][-1]

                    best_model_name = os.path.join(save_model_path, f"best_model_{epoch + 1:04}.pth")
                    print("save model ==>> {}".format(best_model_name))
                    torch.save(vae.state_dict(), best_model_name)


def step_test(test_dataset, device, z_dim, n_hidden, patch_size, save_npy_path, save_model_path):

    cpu = torch.device("cpu")

    img_size = patch_size * patch_size * patch_size

    vae = VariationalAutoEncoder(
        channel=img_size,
        n_hidden=n_hidden,
        z_dim=z_dim,
    )

    print("Loading ==>> {}".format(save_model_path))
    vae.load_state_dict(torch.load(save_model_path))
    vae = vae.to(device)

    vae.eval()

    zs, xs = [], []

    for input_img in tqdm(test_dataset):
        input_img = input_img.to(device)

        z, x = vae(input_img)

        x = x.view(-1, patch_size, patch_size, patch_size)

        zs.append(z.to(cpu).detach().clone().numpy())
        xs.append(x.to(cpu).detach().clone().numpy())

    z = np.concatenate(zs, axis=0)
    x = np.concatenate(xs, axis=0)

    x = post(x)

    file_name_1 = os.path.join(save_npy_path, "z.npy")
    file_name_2 = os.path.join(save_npy_path, "x.npy")
    print("Saving ==>> {}".format(file_name_1))
    print("Saving ==>> {}".format(file_name_2))
    np.save(file_name_1, z)
    np.save(file_name_2, x)

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="./np_record/")
    parser.add_argument('--save_model_path', type=str, default="./model/")
    parser.add_argument('--save_npy_path', type=str, default="./npy/")
    parser.add_argument('--save_sample_path', type=str, default="./sample/")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=200)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default="train", help="[train, test]")

    args = parser.parse_args()
    return args

def main(args):

    check_dir(args.save_model_path)
    check_dir(args.save_npy_path)
    check_dir(args.save_sample_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, valid_dataset, test_dataset = load_data(args.data_path, args.batch_size, args.patch_size)

    if args.mode == "train":
        step_train(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            device=device,
            epochs=args.epochs,
            z_dim=args.z_dim,
            n_hidden=args.n_hidden,
            alpha=args.alpha,
            patch_size=args.patch_size,
            save_model_path=args.save_model_path,
        )

    elif args.mode == "test":
        save_model_path = get_model_name(args.save_model_path)

        step_test(
            test_dataset=test_dataset,
            device=device,
            z_dim=args.z_dim,
            n_hidden=args.n_hidden,
            patch_size=args.patch_size,
            save_npy_path=args.save_npy_path,
            save_model_path=save_model_path,
        )

if __name__ == "__main__":
    args = arg_parser()
    main(args)