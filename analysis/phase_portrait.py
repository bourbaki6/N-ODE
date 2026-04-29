import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from pathlib import Path

from models import NeuralODEClassifier
from data.dataset import get_dataloaders, get_class_names
from utils import get_device, load_checkpoint


def get_hidden_trajectories(model: NeuralODEClassifier, loader, device: torch.device, num_steps: int = 20, max_samples: int = 500) -> dict:
    
    model.eval()
    all_h0, all_h1, all_labels = [], [], []
    all_trajs = []

    ts = np.linspace(0, 1, num_steps)
    n_collected = 0

    with torch.no_grad():
        for images, labels in loader:
            if n_collected >= max_samples:
                break

            images = images.to(device)
            batch_size = images.size(0)

            # Get initial hidden state h(0) through input projection
            x = images.view(batch_size, -1)
            h0 = model.input_proj(x)  # [batch, hidden_dim]

            # Record trajectory by integrating step by step
            traj = [h0.cpu().numpy()]
            h = h0

            # Integrate manually at specified time points
            dt = 1.0 / (num_steps - 1)
            for i in range(1, num_steps):
                t_cur = torch.tensor(ts[i-1], dtype=h.dtype, device=device)
                dh = model.odefunc(t_cur, h)
                h = h + dh * dt
                traj.append(h.cpu().numpy())

            # Stack: traj[i] is [batch, hidden_dim] → transpose to [batch, T, hidden_dim]
            traj_array = np.stack(traj, axis=1)  # [batch, T, hidden]

            all_trajs.append(traj_array)
            all_h0.append(h0.cpu().numpy())
            all_h1.append(h.cpu().numpy())
            all_labels.append(labels.numpy())
            n_collected += batch_size

    return {
        "trajectories": np.concatenate(all_trajs, axis = 0)[:max_samples],
        "h0": np.concatenate(all_h0, axis = 0)[:max_samples],
        "h1": np.concatenate(all_h1, axis = 0)[:max_samples],
        "labels": np.concatenate(all_labels, axis = 0)[:max_samples],
    }


def plot_phase_portrait(
    model: NeuralODEClassifier,
    data: dict,
    class_names: list,
    t: float = 0.0,
    title: str = "Phase Portrait",
    save_path: str = None,
):
   
    hidden_dim = data["h1"].shape[1]

    pca = PCA(n_components=2)
    pca.fit(data["h1"])

    # Project trajectories into PCA space
    trajs_2d = []
    for traj in data["trajectories"]:
        trajs_2d.append(pca.transform(traj))  # [T, 2]

    h0_2d = pca.transform(data["h0"])
    h1_2d = pca.transform(data["h1"])

    # Grid bounds from actual data (with small padding)
    x_min, x_max = h0_2d[:, 0].min() - 0.5, h0_2d[:, 0].max() + 0.5
    y_min, y_max = h0_2d[:, 1].min() - 0.5, h0_2d[:, 1].max() + 0.5

    grid_size = 15
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid_2d = np.stack([gx.ravel(), gy.ravel()], axis=1)  # [G, 2]


    grid_64d = pca.inverse_transform(grid_2d)  #---an approx., [G, 64]---#
    grid_tensor = torch.tensor(grid_64d, dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        dh_64d = model.odefunc(t_tensor, grid_tensor).numpy()  # [G, 64]

    # Project the derivatives into 2D PCA space
    dh_2d = pca.transform(dh_64d) - pca.transform(np.zeros_like(dh_64d))
    u = dh_2d[:, 0].reshape(grid_size, grid_size)
    v = dh_2d[:, 1].reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    colors_10 = cm.tab10.colors

    ax = axes[0]

    speed = np.sqrt(u ** 2 + v ** 2)
    ax.streamplot(
        gx, gy, u, v,
        color = speed, cmap = "Greys", linewidth = 0.8,
        density =0.8, arrowsize = 1.2,
    )

    for cls_idx in range(min(10, len(class_names))):
        mask = data["labels"] == cls_idx
        if mask.sum() == 0:
            continue
        pts = h1_2d[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c = [colors_10[cls_idx]], label = class_names[cls_idx],
            s = 15, alpha = 0.6, edgecolors = "none",
        )

    ax.set_title(f"Vector field at t={t:.1f} + Final states h(1)", fontsize=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax2 = axes[1]

    for cls_idx in range(min(10, len(class_names))):
        mask = np.where(data["labels"] == cls_idx)[0]
        if len(mask) == 0:
            continue

        for sample_idx in mask[:20]:
            traj_2d = trajs_2d[sample_idx]  # [T, 2]
            ax2.plot(
                traj_2d[:, 0], traj_2d[:, 1],
                color=colors_10[cls_idx], linewidth=0.7, alpha=0.4,
            )
            # Mark start (o) and end (x)
            ax2.scatter(traj_2d[0, 0],  traj_2d[0, 1],  s=20, c="white",
                       edgecolors=colors_10[cls_idx], linewidths=1.0, zorder=5)
            ax2.scatter(traj_2d[-1, 0], traj_2d[-1, 1], s=20, marker="x",
                       c=[colors_10[cls_idx]], linewidths=1.5, zorder=5)

    ax2.set_title("ODE Trajectories h(0)→h(1) per class\n(○ = start, × = end)", fontsize=10)
    ax2.set_xlabel(f"PC1")
    ax2.set_ylabel(f"PC2")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=colors_10[i], linewidth=2,
               label = class_names[i] if i < len(class_names) else str(i))
        for i in range(min(10, len(class_names)))
    ]
    ax2.legend(handles = handles, loc = "upper right", fontsize = 7, ncol = 2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description = "Phase Portrait Analysis")
    parser.add_argument("--checkpoint", type = str, required = True)
    parser.add_argument("--config", type = str, default = "configs/mnist.yaml")
    parser.add_argument("--show_untrained",action = "store_true",
                        help = "Also plot portrait for a randomly-initialised model")
    parser.add_argument("--save_dir", type=str, default="analysis/plots")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    device = get_device()

    m_cfg = cfg["model"]
    model = NeuralODEClassifier(
        hidden_dim = m_cfg["hidden_dim"],
        num_classes = m_cfg["num_classes"],
        solver = "euler", 
        num_steps = 20,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device = device)

    _, test_loader = get_dataloaders(
        dataset_name = cfg["dataset"]["name"],
        batch_size = 64,
    )
    class_names = get_class_names(cfg["dataset"]["name"])

    print("Collecting hidden state trajectories")
    data = get_hidden_trajectories(model, test_loader, device, max_samples = 300)

    plot_phase_portrait(
        model, data, class_names,
        t = 0.5,
        title = f"Trained Neural ODE — Phase Portrait ({cfg['dataset']['name'].upper()})",
        save_path = f"{args.save_dir}/phase_portrait_trained.png",
    )

    if args.show_untrained:
        untrained = NeuralODEClassifier(
            hidden_dim = m_cfg["hidden_dim"],
            num_classes = m_cfg["num_classes"],
            solver = "euler", num_steps = 20,
        ).to(device)

        print("Collecting trajectories for untrained model")
        data_untrained = get_hidden_trajectories(untrained, test_loader, device, max_samples = 300)

        plot_phase_portrait(
            untrained, data_untrained, class_names,
            t = 0.5,
            title = "UNTRAINED Neural ODE — Phase Portrait (random weights)",
            save_path = f"{args.save_dir}/phase_portrait_untrained.png",
        )
        print("\n Compare the two portraits:")
        print("\n Untrained: uniform/random vector field, overlapping class clouds")
        print("\n Trained:   structured attractors, separated class trajectories")


if __name__ == "__main__":
    main()