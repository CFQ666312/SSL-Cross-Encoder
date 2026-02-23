# src/train_pretrain.py
import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import tqdm

# ----------------
# Your project imports
# -----------------------
# Models (you can also do: from src.models import Cross_Sim, AE, LSP, ...
from models import *  # noqa

# Dataset wrapper should expose LongitudinalData with .trainLoader/.valLoader/.testLoader
from data import LongitudinalData  # adjust if your filename differs

# Utilities: you implement these under src/utils/
from utils.io import load_config_yaml, save_config_yaml, save_checkpoint, load_checkpoint_by_key
from utils.meter import save_result_stat


def set_seed(seed: int = 10) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_run_name(cfg: Dict[str, Any]) -> str:
    """
    Decide run name / timelabel.
    If user provides ckpt_timelabel AND (test or continue_train), reuse it.
    Otherwise generate a timestamp string.
    """
    if cfg.get("ckpt_timelabel") and (cfg.get("phase") == "test" or cfg.get("continue_train") is True):
        return str(cfg["ckpt_timelabel"])

    t = time.localtime(time.time())
    return f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}"


def prepare_ckpt_dir(cfg: Dict[str, Any], run_name: str) -> str:
    """
    Create checkpoint directory under repo-relative path.
    Example: ./ckpt/<dataset>/<model>/<run_name>/
    """
    root = cfg.get("ckpt_root", "ckpt")
    dataset_tag = cfg.get("dataset_name", "DATASET")
    model_name = cfg.get("model_name", "MODEL")

    ckpt_path = os.path.join(root, dataset_tag, model_name, run_name)
    os.makedirs(ckpt_path, exist_ok=True)
    return ckpt_path


def maybe_reload_yaml(cfg: Dict[str, Any], ckpt_path: str) -> Dict[str, Any]:
    """
    If ckpt_path exists and cfg['load_yaml'] is True, load config.yaml from ckpt dir and merge.
    Keep runtime keys (phase/gpu/continue_train/ckpt_name) from current cfg.
    """
    yaml_path = os.path.join(ckpt_path, "config.yaml")
    if os.path.exists(yaml_path) and cfg.get("load_yaml", False):
        flag, cfg_load = load_config_yaml(yaml_path)
        if flag:
            skip = {"phase", "gpu", "continue_train", "ckpt_name"}
            for k, v in cfg_load.items():
                if k in skip:
                    continue
                cfg[k] = v
        else:
            save_config_yaml(ckpt_path, cfg)
    else:
        # if not exists, write the current cfg
        save_config_yaml(ckpt_path, cfg)

    return cfg


def build_dataloaders(cfg: Dict[str, Any]):
    data = LongitudinalData(
        cfg["dataset_name"],
        cfg["data_path"],
        img_file_name=cfg.get("img_file_name"),
        noimg_file_name=cfg.get("noimg_file_name"),
        subj_list_postfix=cfg.get("subj_list_postfix"),
        data_type=cfg.get("data_type", "default"),
        batch_size=cfg.get("batch_size", 8),
        num_fold=cfg.get("num_fold", 5),
        fold=cfg.get("fold", 0),
        shuffle=cfg.get("shuffle", True),
    )
    return data.trainLoader, data.valLoader, data.testLoader


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    name = cfg["model_name"]

    if name == "LSP":
        # adjust your LSP signature if needed
        model = LSP(
            latent_size=cfg.get("latent_size", 1024),
            num_neighbours=cfg.get("num_neighbours", 3),
            agg_method=cfg.get("agg_method", "gaussian"),
            device=cfg.get("device", None),
        )
    elif name == "AE":
        model = AE()
    elif name == "Cross_Sim":
        model = Cross_Sim(
            selection=cfg.get("selection", 0.75),
            temperature=cfg.get("temperature", 0.5),
        )
    elif name == "Cross_Sim_ViT":
        model = Cross_Sim_ViT(
            img_size=tuple(cfg.get("img_size", [64, 64, 64])),
            patch_size=tuple(cfg.get("patch_size", [8, 8, 8])),
            embed_dim=cfg.get("embed_dim", 1024),
            depth=cfg.get("vit_depth", 12),
            num_heads=cfg.get("vit_heads", 16),
            dynamic_ratio=cfg.get("selection", 0.25),
            temperature=cfg.get("temperature", 0.5),
        )
    else:
        raise ValueError(f"Unsupported model_name: {name}")

    return model


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
        amsgrad=True,
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.get("lr_factor", 0.1),
        patience=cfg.get("lr_patience", 5),
        min_lr=cfg.get("min_lr", 1e-5),
    )
    return optimizer, scheduler


def compute_losses(cfg: Dict[str, Any], model, img1, img2, out) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    out = (zs, zd, recons, img) for Cross_Sim family.
    """
    zs, zd, recons, img = out

    loss = torch.tensor(0.0, device=img1.device)
    loss_dict = {"all": 0.0, "recon": 0.0, "Grad": 0.0, "Contrastive": 0.0}

    # recon
    if cfg.get("lambda_recon", 0.0) > 0:
        loss_recon = 0.5 * (model.compute_recon_loss(img1, recons[0]) + model.compute_recon_loss(img2, recons[1]))
        loss = loss + cfg["lambda_recon"] * loss_recon
        loss_dict["recon"] = float(loss_recon.detach().cpu())
    else:
        loss_recon = None

    # contrastive (on static or dynamic depends on your implementation; here matches your old call)
    if cfg.get("lambda_Sim", 0.0) > 0:
        loss_contrast = model.Contrastiveloss(zs[0], zs[1], zs[0].shape[0])
        loss = loss + cfg["lambda_Sim"] * loss_contrast
        loss_dict["Contrastive"] = float(loss_contrast.detach().cpu())

    # input-gradient reg
    if cfg.get("lambda_Grad", 0.0) > 0:
        # your model.compute_img_gradients returns scalar
        loss_grad = model.compute_img_gradients(img, zd[0], zd[1])
        loss = loss + cfg["lambda_Grad"] * loss_grad
        loss_dict["Grad"] = float(loss_grad.detach().cpu())

    loss_dict["all"] = float(loss.detach().cpu())
    return loss, loss_dict


@torch.no_grad()
def evaluate(cfg: Dict[str, Any], model: torch.nn.Module, loader, phase: str, save_res: bool = False) -> Dict[str, float]:
    model.eval()

    # lazily import only when needed
    if save_res:
        import h5py  # noqa

        res_dir = os.path.join(cfg["ckpt_path"], f"result_{phase}")
        os.makedirs(res_dir, exist_ok=True)
        h5_path = os.path.join(res_dir, "results.h5")
        if os.path.exists(h5_path):
            os.remove(h5_path)

        img1_list, img2_list = [], []
        recon1_list, recon2_list = [], []
        z1_list, z2_list = [], []
        interval_list, label_list = [], []
    else:
        h5_path = None

    agg = {"all": 0.0, "recon": 0.0, "Grad": 0.0, "Contrastive": 0.0}
    n = 0

    for sample in tqdm.tqdm(loader, desc=f"eval:{phase}"):
        img1 = sample["img1"].to(cfg["device"], dtype=torch.float).unsqueeze(1)
        img2 = sample["img2"].to(cfg["device"], dtype=torch.float).unsqueeze(1)
        label = sample.get("label", None)
        interval = sample.get("interval", None)

        out = model(img1, img2)

        # compute losses (same as train, but no backward)
        loss, loss_dict = compute_losses(cfg, model, img1, img2, out)

        for k in agg:
            agg[k] += loss_dict.get(k, 0.0)
        n += 1

        if save_res:
            zs, zd, recons, _ = out
            img1_list.append(img1.detach().cpu().numpy())
            img2_list.append(img2.detach().cpu().numpy())
            recon1_list.append(recons[0].detach().cpu().numpy())
            recon2_list.append(recons[1].detach().cpu().numpy())
            z1_list.append(zs[0].detach().cpu().numpy())
            z2_list.append(zs[1].detach().cpu().numpy())
            if interval is not None:
                interval_list.append(interval.detach().cpu().numpy())
            if label is not None:
                label_list.append(label.detach().cpu().numpy())

    for k in agg:
        agg[k] = agg[k] / max(1, n)

    if save_res:
        import h5py  # noqa
        img1_arr = np.concatenate(img1_list, axis=0)
        img2_arr = np.concatenate(img2_list, axis=0)
        recon1_arr = np.concatenate(recon1_list, axis=0)
        recon2_arr = np.concatenate(recon2_list, axis=0)
        z1_arr = np.concatenate(z1_list, axis=0)
        z2_arr = np.concatenate(z2_list, axis=0)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("img1", data=img1_arr)
            f.create_dataset("img2", data=img2_arr)
            f.create_dataset("recon1", data=recon1_arr)
            f.create_dataset("recon2", data=recon2_arr)
            f.create_dataset("z1", data=z1_arr)
            f.create_dataset("z2", data=z2_arr)
            if interval_list:
                f.create_dataset("interval", data=np.concatenate(interval_list, axis=0))
            if label_list:
                f.create_dataset("label", data=np.concatenate(label_list, axis=0))

    return agg


def train(cfg: Dict[str, Any], model, optimizer, scheduler, train_loader, val_loader, start_epoch: int = -1):
    global_iter = 0
    best_metric = float("inf")

    for epoch in range(start_epoch + 1, cfg["epochs"]):
        model.train()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{cfg['epochs']-1} | lr={cur_lr:.6f}")

        # meters
        agg = {"all": 0.0, "recon": 0.0, "Grad": 0.0, "Contrastive": 0.0}
        n = 0

        for it, sample in enumerate(train_loader):
            global_iter += 1

            img1 = sample["img1"].to(cfg["device"], dtype=torch.float).unsqueeze(1)
            img2 = sample["img2"].to(cfg["device"], dtype=torch.float).unsqueeze(1)

            # (optional) drop too-small batch like your old code
            if img1.shape[0] <= cfg.get("batch_size", 8) // 2:
                continue

            out = model(img1, img2)
            loss, loss_dict = compute_losses(cfg, model, img1, img2, out)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # (optional) NaN guard
            for _, p in model.named_parameters():
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    raise FloatingPointError("Non-finite gradient detected.")

            optimizer.step()

            for k in agg:
                agg[k] += loss_dict.get(k, 0.0)
            n += 1

            if (it % cfg.get("log_every", 10)) == 0:
                print(
                    f"  iter {it:04d} | "
                    f"loss={loss_dict['all']:.4f} "
                    f"recon={loss_dict['recon']:.4f} "
                    f"grad={loss_dict['Grad']:.4f} "
                    f"sim={loss_dict['Contrastive']:.4f}"
                )

        # epoch avg
        for k in agg:
            agg[k] = agg[k] / max(1, n)

        save_result_stat(agg, cfg, info=f"train_epoch[{epoch:03d}]")
        print("train:", agg)

        # validation
        val_stat = evaluate(cfg, model, val_loader, phase="val", save_res=False)
        save_result_stat(val_stat, cfg, info="val")
        print("val:", val_stat)

        monitor_metric = val_stat["all"]
        scheduler.step(monitor_metric)

        # checkpoint
        is_best = monitor_metric <= best_metric
        best_metric = min(best_metric, monitor_metric)

        state = {
            "epoch": epoch,
            "monitor_metric": monitor_metric,
            "stat": val_stat,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict(),
        }
        save_checkpoint(state, is_best, cfg["ckpt_path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to yaml config")
    args = parser.parse_args()

    # seed
    set_seed(10)

    # load config
    _, cfg = load_config_yaml(args.config)

    # device
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run name / ckpt dir
    run_name = make_run_name(cfg)
    cfg["ckpt_timelabel"] = run_name
    cfg["ckpt_path"] = prepare_ckpt_dir(cfg, run_name)

    # merge yaml if needed + always save a copy
    cfg = maybe_reload_yaml(cfg, cfg["ckpt_path"])
    print("Config:\n", cfg)

    # data
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # model
    model = build_model(cfg).to(cfg["device"])

    # optim
    optimizer, scheduler = build_optimizer(cfg, model)

    # resume
    start_epoch = -1
    if cfg.get("continue_train", False) or cfg.get("phase") == "test":
        ckpt_name = cfg.get("ckpt_name", "checkpoint_best.pth")
        (optimizer, scheduler, model), start_epoch = load_checkpoint_by_key(
            [optimizer, scheduler, model],
            cfg["ckpt_path"],
            keys=["optimizer", "scheduler", "model"],
            device=cfg["device"],
            ckpt_name=ckpt_name,
        )
        print("Resumed from epoch:", start_epoch)
        print("Starting lr:", optimizer.param_groups[0]["lr"])

    # run
    if cfg.get("phase", "train") == "train":
        train(cfg, model, optimizer, scheduler, train_loader, val_loader, start_epoch=start_epoch)
    else:
        # save results for test
        _ = evaluate(cfg, model, train_loader, phase="train", save_res=True)
        stat = evaluate(cfg, model, test_loader, phase="test", save_res=True)
        print("test:", stat)


if __name__ == "__main__":
    main()
