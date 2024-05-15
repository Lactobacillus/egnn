from qm9 import dataset
from qm9.models import EGNN
import torch
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils
import utils
import json

import os
import wandb

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='qm9/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='learning rate')
parser.add_argument('--attention', type=int, default=1, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--property', type=str, default='U0', metavar='N',
                    help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')
parser.add_argument('--dataset_paper', type=str, default="cormorant", metavar='N',
                    help='cormorant, lie_conv')
parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                    help='node_attr or not')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')
parser.add_argument('--use_force', action = 'store_true')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

utils.makedir(args.outf)
utils.makedir(args.outf + "/" + args.exp_name)

dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)
# compute mean and mean absolute deviation
meann, mad = qm9_utils.compute_mean_mad(dataloaders, args.property)

model = EGNN(in_node_nf=15, in_edge_nf=0, hidden_nf=args.nf, device=device, n_layers=args.n_layers, coords_weight=1.0,
             attention=args.attention, node_attr=args.node_attr)

print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()


def train(epoch, loader, partition='train'):
    lr_scheduler.step()
    res = {'energy': 0, 'force': 0, 'loss': 0, 'counter': 0, 'loss_arr':[], 'energy_arr':[], 'force_arr':[]}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        energy = data[args.property].to(device, dtype)

        atom_positions.requires_grad = True
        atom_positions.retain_grad()

        pred_energy = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        energy_per_atom = energy / n_nodes
        pred_energy_per_atom = pred_energy / n_nodes

        if args.use_force:

            pred_force = -torch.autograd.grad(
                            pred_energy,
                            atom_positions,
                            grad_outputs = torch.ones_like(pred_energy),
                            create_graph = True)[0].detach()

            pred_force_per_atom = pred_force / n_nodes

        else:

            pred_force = torch.zeros_like(atom_positions)
            pred_force_per_atom = torch.zeros_like(atom_positions)

        if partition == 'train':
            
            energy_loss = loss_l1(pred_energy_per_atom, energy_per_atom)
            force_loss = loss_l1(pred_force_per_atom, torch.zeros_like(pred_force_per_atom))

            loss = energy_loss + 0.3 * force_loss

            loss.backward()
            optimizer.step()

        else:

            energy_loss = loss_l1(pred_energy_per_atom, energy_per_atom)
            force_loss = loss_l1(pred_force_per_atom, torch.zeros_like(pred_force_per_atom))

            loss = energy_loss + 0.3 * force_loss

        res['loss'] += loss.item() * batch_size
        res['energy'] += energy_loss.item() * batch_size
        res['force'] += force_loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        res['energy_arr'].append(energy_loss.item())
        res['force_arr'].append(force_loss.item())

        if partition == 'train':

            wandb_log_dict = {'train_iter/energy_mae_loss_meV': 1000 * energy_loss.item(),
                            'train_iter/force_mae_loss_meV_angstrom': 1000 * force_loss.item(),
                            'train_iter/total_mae_loss': 1000 * loss.item()}
            wandb.log(wandb_log_dict)

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % args.log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
            print(prefix + "            energy_loss meV %.4f \t force_loss meV %.4f" % (1000 * energy_loss.item(), 1000 * force_loss.item()))
    return res['loss'] / res['counter']


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'energy_per_atom': [], 'force_per_atom':[], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    wandb.init(project = 'stable-material-diffusion_regression',
        name = args.exp_name,
        entity = 'feedstock_opt',
        dir = os.path.join(args.outf, args.exp_name))
    wandb.config.update(args)

    for epoch in range(0, args.epochs):
        train(epoch, dataloaders['train'], partition='train')
        if epoch % args.test_interval == 0:
            val_loss = train(epoch, dataloaders['valid'], partition='valid')
            test_loss = train(epoch, dataloaders['test'], partition='test')
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)
            res['energy_per_atom'].append(test_loss)
            res['force_per_atom'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))

        json_object = json.dumps(res, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)

    wandb.finish(0)
