import sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models.classifier import NeuralODEClassifier
from baseline.resnet import ResNetBaseline
from data.dataset import get_dataloaders, get_class_names, DATASET_STATS
from utils import get_device, save_checkpoint, MetricsLogger, measure_nfe


OUT = Path("outputs")
CKPTS = OUT / "checkpoints"
LOGS = OUT / "logs"
FIGS = OUT / "figures"
for d in [CKPTS, LOGS, FIGS]:
    d.mkdir(parents = True, exist_ok = True)

def ok(msg): print(f"{msg}")
def hdr(msg):
    print(f"\n{'─'*55}")
    print(f"  {msg}")
    print(f"{'─'*55}")

COLORS = {"ode_euler": "#C0392B", "ode_rk4": "#1A5276", "resnet6": "#117A65"}
LABELS = {"ode_euler": "ODE (Euler)", "ode_rk4": "ODE (RK4)", "resnet6": "ResNet-6"}


def train_model(model, run_name, epochs, device):
    train_loader, test_loader = get_dataloaders(
        "mnist", batch_size = 128, num_workers = 0
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = epochs, eta_min  = 1e-5
    )
    criterion = nn.NLLLoss()
    logger = MetricsLogger(str(LOGS), run_name)
    best_acc = 0.0
    best_path = str(CKPTS / f"{run_name}_best.pt")

    for epoch in range(1, epochs + 1):

        model.train()
        tl = tc = tt = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            lp = model(imgs)
            loss = criterion(lp, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            tc += (lp.argmax(1) == lbls).sum().item()
            tt += imgs.size(0)

        model.eval()
        el = ec = et = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                lp  = model(imgs)
                el += criterion(lp, lbls).item() * imgs.size(0)
                ec += (lp.argmax(1) == lbls).sum().item()
                et += imgs.size(0)

        tr_acc  = tc / tt; te_acc = ec / et
        tr_loss = tl / tt; te_loss = el / et
        nfe = measure_nfe(model, test_loader, device, num_batches = 3) \
              if hasattr(model, "nfe") else None

        kw = dict(train_loss = tr_loss, test_loss = te_loss,
                  train_acc = tr_acc, test_acc = te_acc,
                  lr = scheduler.get_last_lr()[0])
        if nfe is not None:
            kw["nfe"] = nfe
        logger.log(epoch = epoch, **kw)
        scheduler.step()

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint(model, optimizer, epoch, kw, best_path)

        nfe_s = f"  NFE={nfe:.0f}" if nfe else ""
        print(f"ep {epoch:3d}/{epochs}  "
              f"loss={tr_loss:.4f}  train={tr_acc:.3f}  "
              f"test={te_acc:.3f}{nfe_s}")

    logger.save()
    return {
        "run_name": run_name,
        "best_acc": best_acc,
        "ckpt": best_path,
        "records": logger.records,
        "params": sum(p.numel() for p in model.parameters()),
    }


def _ax_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)


def plot_training_curves(all_results):
    fig, axes = plt.subplots(2, 2, figsize = (11, 7))
    axes = axes.flat
    panels = [
        ("\n train_loss", "\n Training Loss", "\n Loss"),
        ("\n test_acc", "\n Test Accuracy", "\n Accuracy"),
        ("\n nfe", "NFE over Epochs\n(lower = simpler dynamics)", "NFE"),
        ("\n lr", "\n Learning Rate", "\n LR"),
    ]
    for ax, (key, title, ylabel) in zip(axes, panels):
        for rname, r in all_results.items():
            vals = [rec[key] for rec in r["records"] if key in rec]
            eps = [rec["epoch"] for rec in r["records"] if key in rec]
            if not vals: continue
            ax.plot(eps, vals, color = COLORS.get(rname, "gray"),
                    label = LABELS.get(rname, rname), linewidth = 1.8)
        ax.set_title(title, fontsize = 10); ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel); ax.legend(fontsize = 8); _ax_clean(ax)

    fig.suptitle("\n Training Diagnostics", fontsize = 13, fontweight = "bold")
    plt.tight_layout()
    plt.savefig(str(FIGS/"training_curves.png"), dpi = 150, bbox_inches = "tight")
    plt.close()
    ok("training_curves.png")


def plot_solver_comparison(ckpt_path, device):
    _, test_loader = get_dataloaders("mnist", batch_size = 128, num_workers = 0)
    model = NeuralODEClassifier(hidden_dim = 64, num_classes = 10,
                                solver = "euler", num_steps = 10)
    ckpt = torch.load(ckpt_path, map_location = device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    rows = []
    for solver in ["euler", "rk4"]:
        for steps in [2, 4, 6, 8, 10, 15, 20]:
            model.set_solver(solver, num_steps = steps)
            correct = total = 0
            nfes = []; times = []
            with torch.no_grad():
                for i, (imgs, lbls) in enumerate(test_loader):
                    if i >= 10: break
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    t0 = time.perf_counter()
                    lp = model(imgs)
                    times.append((time.perf_counter()-t0)*1000/imgs.size(0))
                    correct += (lp.argmax(1)==lbls).sum().item()
                    total += lbls.size(0)
                    nfes.append(model.nfe)
            rows.append(dict(solver = solver, steps = steps,
                             acc = correct/total*100,
                             nfe = np.mean(nfes), lat = np.mean(times)))

    euler_r = [r for r in rows if r["solver"]=="euler"]
    rk4_r = [r for r in rows if r["solver"]=="rk4"]

    fig, axes = plt.subplots(1, 3, figsize = (13, 4))
    for ax, xkey, xlabel, ykey, ylabel, title in [
        (axes[0], "nfe","NFE", "acc", "Accuracy (%)", "Accuracy vs NFE"),
        (axes[1], "lat", "Latency (ms/img)", "acc", "Accuracy (%)", "Accuracy vs Latency"),
        (axes[2], "steps", "num_steps","nfe", "NFE", "NFE vs Steps"),
    ]:
        for grp, c, lbl in [(euler_r, COLORS["ode_euler"], "Euler"),
                             (rk4_r,  COLORS["ode_rk4"], "RK4")]:
            ax.plot([r[xkey] for r in grp], [r[ykey] for r in grp],
                    "o-", color = c, lw = 1.8, markersize = 5, label = lbl)
        ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel); ax.legend(fontsize=8); _ax_clean(ax)

    fig.suptitle("Solver Comparison — Same Model, Different Solvers",
                 fontsize = 12, fontweight = "bold")
    plt.tight_layout()
    plt.savefig(str(FIGS/"solver_comparison.png"), dpi = 150, bbox_inches = "tight")
    plt.close()
    ok("solver_comparison.png")


def plot_phase_portrait(ckpt_path, device):
    _, test_loader = get_dataloaders("mnist", batch_size = 64, num_workers = 0)
    cnames = get_class_names("mnist")
    C10 = plt.cm.tab10.colors

    def collect(mdl, n=300):
        mdl.eval()
        h0s, h1s, ls = [], [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                if sum(len(x) for x in h0s) >= n: break
                imgs = imgs.to(device)
                x  = imgs.view(imgs.size(0), -1)
                h0 = mdl.input_proj(x)
                h1 = mdl.ode_block(h0)
                h0s.append(h0.cpu().numpy())
                h1s.append(h1.cpu().numpy())
                ls.append(lbls.numpy())
        return (np.concatenate(h0s)[:n],
                np.concatenate(h1s)[:n],
                np.concatenate(ls)[:n])

    trained = NeuralODEClassifier(hidden_dim = 64, num_classes = 10,
                                  solver = "euler", num_steps = 10)
    ckpt = torch.load(ckpt_path, map_location = device)
    trained.load_state_dict(ckpt["model_state_dict"])
    trained = trained.to(device)
    untrained = NeuralODEClassifier(hidden_dim = 64, num_classes = 10,
                                    solver = "euler", num_steps = 10).to(device)

    fig, axes = plt.subplots(1, 2, figsize = (13, 5.5))
    for ax, mdl, title in [
        (axes[0], untrained, "Untrained (random weights)"),
        (axes[1], trained, "Trained — learned vector field"),
    ]:
        h0, h1, lbls = collect(mdl)
        pca = PCA(n_components=2)
        h1_2d = pca.fit_transform(h1)
        h0_2d = pca.transform(h0)

        for cls in range(10):
            mask = lbls == cls
            if mask.sum() == 0: continue
            ax.scatter(h1_2d[mask,0], h1_2d[mask,1],
                       c=[C10[cls]], s = 12, alpha = 0.7,
                       edgecolors = "none", label = cnames[cls])
            for i in np.where(mask)[0][:5]:
                ax.annotate("",
                    xy = (h1_2d[i,0], h1_2d[i,1]),
                    xytext = (h0_2d[i,0], h0_2d[i,1]),
                    arrowprops = dict(arrowstyle="-|>", lw = 0.6,
                                   color = C10[cls], alpha = 0.35,
                                   mutation_scale = 7))

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.legend(fontsize=6, ncol=2); _ax_clean(ax)

    fig.suptitle("Phase Portrait — h(0) -> h(1) trajectories in PCA space",
                 fontsize = 12, fontweight = "bold")
    plt.tight_layout()
    plt.savefig(str(FIGS/"phase_portrait.png"), dpi = 150, bbox_inches = "tight")
    plt.close()
    ok("phase_portrait.png")


def plot_summary(all_results):
    keys = [k for k in ["ode_euler","ode_rk4","resnet6"] if k in all_results]
    labels = [LABELS[k] for k in keys]
    accs = [all_results[k]["best_acc"]*100 for k in keys]
    params = [all_results[k]["params"]/1000  for k in keys]
    colors = [COLORS[k] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize = (10, 4))

    bars = axes[0].bar(labels, accs, color = colors, edgecolor = "white", width = 0.5)
    axes[0].bar_label(bars, fmt = "%.2f%%", padding = 3, fontsize = 9)
    axes[0].set_title("Test Accuracy", fontsize = 11)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(max(0, min(accs)-5), min(max(accs)+5, 100))
    _ax_clean(axes[0])

    bars2 = axes[1].bar(labels, params, color = colors, edgecolor = "white", width = 0.5)
    axes[1].bar_label(bars2, fmt = "%.0fK", padding = 3, fontsize = 9)
    axes[1].set_title("Parameter Count", fontsize = 11)
    axes[1].set_ylabel("Parameters (K)")
    _ax_clean(axes[1])

    if "ode_rk4" in all_results and "resnet6" in all_results:
        ratio = all_results["ode_rk4"]["params"] / all_results["resnet6"]["params"] * 100
        axes[1].annotate(f"ODE uses {ratio:.0f}%\nof ResNet-6 params",
                         xy = (0.97, 0.97), xycoords = "axes fraction",
                         ha = "right", va = "top", fontsize = 8,
                         bbox = dict(boxstyle  = "round", facecolor = "#EEF2F7",
                                   edgecolor = "#AAAAAA", alpha = 0.9))

    fig.suptitle("Summary — Neural ODE vs ResNet Baseline",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(FIGS/"summary.png"), dpi = 150, bbox_inches = "tight")
    plt.close()
    ok("summary.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke",  action="store_true",
                   help="3-epoch run to verify everything works")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--solver", type=str, default=None,
                   help="euler or rk4  (default: both)")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 3
        solvers = ["euler"]
    else:
        solvers = [args.solver] if args.solver else ["euler", "rk4"]

    print(f"\n{'═'*55}")
    print(f" Neural ODE Pipeline")
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" epochs = {args.epochs}  solvers={solvers}")
    print(f"{'═'*55}")

    device = get_device()

    hdr("Data")
    train_loader, _ = get_dataloaders("mnist", batch_size = 128, num_workers = 0)
    imgs, lbls = next(iter(train_loader))
    assert imgs.shape[1:] == torch.Size([1, 28, 28])
    assert imgs.min().item() < 0, "Normalisation not applied"
    ok(f"MNIST  shape={tuple(imgs.shape)}  "
       f"range=[{imgs.min():.2f}, {imgs.max():.2f}]")

    stats = DATASET_STATS["mnist"]
    display = (imgs[:16].clone()*stats["std"][0]+stats["mean"][0]).clamp(0,1)
    cnames = get_class_names("mnist")
    fig, axes = plt.subplots(2, 8, figsize = (12, 3.5))
    for i, ax in enumerate(axes.flat):
        if i >= 16: ax.set_visible(False); continue
        ax.imshow(display[i].squeeze().numpy(), cmap = "gray")
        ax.set_title(cnames[lbls[i].item()], fontsize = 7)
        ax.axis("off")
    plt.suptitle("MNIST samples", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(OUT/"data_grid.png"), dpi = 120, bbox_inches = "tight")
    plt.close()
    ok("data_grid.png")

    all_results = {}

    for solver in solvers:
        hdr(f"Neural ODE  solver = {solver}  epochs = {args.epochs}")
        model  = NeuralODEClassifier(hidden_dim = 64, num_classes = 10,
                                     solver=solver, num_steps = 10)
        rname = f"ode_{solver}"
        r = train_model(model, rname, args.epochs, device)
        r["model_type"] = "NeuralODE"
        r["solver"] = solver
        all_results[rname] = r
        ok(f"best_acc={r['best_acc']*100:.2f}%  params={r['params']:,}")

    hdr(f"ResNet-6  epochs={args.epochs}")
    resnet = ResNetBaseline(hidden_dim = 64, num_blocks = 6, num_classes = 10)
    r  = train_model(resnet, "resnet6", args.epochs, device)
    r["model_type"] = "ResNet"
    r["solver"] = "N/A"
    all_results["resnet6"] = r
    ok(f"best_acc={r['best_acc']*100:.2f}%  params = {r['params']:,}")

 
    hdr("Figures")
    plot_training_curves(all_results)
    plot_summary(all_results)

    ode_key = "ode_rk4" if "ode_rk4" in all_results else "ode_euler"
    try:
        plot_solver_comparison(all_results[ode_key]["ckpt"], device)
    except Exception as e:
        print(f"  ! solver comparison skipped: {e}")
    try:
        plot_phase_portrait(all_results[ode_key]["ckpt"], device)
    except Exception as e:
        print(f"  ! phase portrait skipped: {e}")

    #
    hdr("Results")
    print(f"\n  {'Model':<20} {'Accuracy':>10} {'Params':>10}")
    print(f"  {'─'*42}")
    
    for rname, r in all_results.items():
        lbl = LABELS.get(rname, rname)
        print(f"  {lbl:<20} {r['best_acc']*100:>9.2f}%  {r['params']:>9,}")

    print(f"\n  Outputs -> {OUT.resolve()}")
    print(f"  Figures -> {FIGS.resolve()}\n")


if __name__ == "__main__":
    main()