import torch
import numpy as np
from stlcgpp.formula import *
import matplotlib.pyplot as plt
import timeit
import statistics
import pickle
import sys

# from matplotlib import rc
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(5, 64),
                                       torch.nn.Tanh(),
                                       torch.nn.Linear(64, 64),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(64, 1))


    def forward(self, signal: torch.Tensor):
        return self.net(signal).squeeze(-1)



if __name__ == "__main__":

    args = sys.argv[1:]
    bs = int(args[0])
    max_T = int(args[1])
    device = str(args[2])
    formula_type = str(args[3])
    if len(args) > 4:
        extra = str(args[4])
    else:
        extra = ""

    filename = "../results/timing_test_pytorch_bs_%i_maxT_%i_%s_%s_%s"%(bs, max_T, device, formula_type, extra)


    signal_dim = 5
    predicate_func = Net()

    predicate = Predicate('x', predicate_func)
    interval = None
    phi = GreaterThan(predicate, 2.)
    psi = LessThan(predicate, 4.)


    if formula_type == "always":
        mask = Always(And(phi, psi), interval=interval).to(device)
        recurrent = AlwaysRecurrent(And(phi, psi), interval=interval).to(device)

    elif formula_type == "eventually_always":
        mask = Eventually(Always(And(phi, psi), interval=interval)).to(device)
        recurrent = EventuallyRecurrent(AlwaysRecurrent(And(phi, psi), interval=interval)).to(device)
    elif formula_type == "until":
        mask = Until(phi, psi, interval=interval).to(device)
        recurrent = UntilRecurrent(phi, psi, interval=interval).to(device)
    else:
        raise NotImplementedError


    def foo(signal):
        return mask(signal).mean()

    def mask_(signal):
        return torch.vmap(foo)(signal)

    def grad_mask(signal):
        return torch.vmap(torch.func.grad(foo))(signal)

    def goo(signal):
        return recurrent(signal).mean()

    def recurrent_(signal):
        return torch.vmap(recurrent)(signal)

    def grad_recurrent(signal):
        return torch.vmap(torch.func.grad(goo))(signal)


    # Number of loops per run
    loops = 5
    # Number of runs
    runs = 5
    T = 2

    data = {}
    Ts = []

    # functions = ["mask_", "grad_mask", "mask_jit", "grad_mask_jit", "recurrent_", "grad_recurrent", "recurrent_jit", "grad_recurrent_jit"]
    functions = ["mask_", "recurrent_", "grad_mask", "grad_recurrent"]
    data["functions"] = functions
    data["runs"] = runs
    data["loops"] = loops


    while T <= max_T:
        Ts.append(T)
        data['Ts'] = Ts
        print("running ", T)
        signal = torch.rand([bs, T, signal_dim]).to(device)
        times_list = []
        data[str(T)] = {}
        for f in functions:
            print("timing ", f)
            timeit.repeat(f + "(signal)", globals=globals(), repeat=1, number=1)
            times = timeit.repeat(f + "(signal)", globals=globals(), repeat=runs, number=loops)
            times_list.append(times)
            print("timing: ", statistics.mean(times), statistics.stdev(times))
            data[str(T)][f] = times
            with open(filename + '.pkl', 'wb') as f:
                pickle.dump(data, f)

        T *= 2

    # means = []
    # stds = []
    # for k in data.keys():
    #     if k in ["Ts", "functions", "loops", "runs"]:
    #         continue
    #     mus = []
    #     sts = []
    #     for f in data[k].keys():
    #         mus.append(statistics.mean(data[k][f])/data["loops"])
    #         sts.append(statistics.stdev(data[k][f])/data["loops"])

    #     means.append(mus)
    #     stds.append(sts)
    # means = np.array(means)
    # stds = np.array(stds)

    # fontsize = 14

    # plt.figure(figsize=(10,5))
    # plt.plot(data["Ts"], means * 1E3)
    # for (m,s) in zip(means.T, stds.T):
    #     plt.fill_between(data["Ts"], (m - s) * 1E3, (m + s) * 1E3, alpha=0.3)
    # plt.yscale("log")
    # plt.legend(functions, fontsize=fontsize-2)
    # plt.grid()
    # plt.xlabel("Signal length", fontsize=fontsize)
    # plt.ylabel("Computation time [ms]", fontsize=fontsize)
    # plt.title("Pytorch " + str(device), fontsize=fontsize+2)
    # plt.tight_layout()

    # plt.savefig(filename + ".png", dpi=200, transparent=True)


