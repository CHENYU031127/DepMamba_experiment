import argparse
import os
import yaml

import wandb
import torch
from tqdm import tqdm

from models import DepMamba
from datasets import get_dvlog_dataloader, get_lmvd_dataloader

CONFIG_PATH = "./config/config.yaml"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = v.lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: True/False")
    
def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model."
    )
    # arguments whose default values are in config.yaml
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("--use_local_alignment", type=str2bool)
    parser.add_argument("--use_global_alignment", type=str2bool)
    parser.add_argument("--lambda_local_alignment", type=float)
    parser.add_argument("--lambda_global_alignment", type=float)
    
    parser.add_argument("-g", "--gpu", type=str, default="0", help="GPU id(s), e.g. '0' or '0,1' or 'cuda:0,1' or 'cpu'")
    
    parser.add_argument("-wdb", "--if_wandb", type=str2bool, default=False)
    parser.add_argument("-tqdm", "--tqdm_able", type=str2bool)
    parser.add_argument("-tr", "--train", type=str2bool)
    parser.set_defaults(**config)
    args = parser.parse_args()
    return args

# 解析 --gpu 字符串为绝对 GPU 编号列表；支持 cpu  # <<<
def _parse_gpu_arg(gpu_arg: str):
    s = str(gpu_arg).strip().lower()
    if s in ("cpu", "none", "-1", ""):
        return None  # 表示使用 CPU
    s = s.replace("cuda:", "")
    ids = [int(x) for x in s.split(",") if x != ""]
    return ids


def _unwrap_model(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def _format_for_path(value):
    text = str(value)
    return (
        text.replace("-", "m")
        .replace(".", "p")
        .replace("/", "_")
        .replace("\\", "_")
    )


def build_experiment_name(args):
    local_tag = "local1" if args.use_local_alignment else "local0"
    global_tag = "global1" if args.use_global_alignment else "global0"
    lambda_local = _format_for_path(args.lambda_local_alignment)
    lambda_global = _format_for_path(args.lambda_global_alignment)
    return (
        f"{args.dataset}_{args.model}_"
        f"{local_tag}_ll{lambda_local}_"
        f"{global_tag}_lg{lambda_global}"
    )


def train_epoch(
    net, train_loader, loss_fn, optimizer, device, 
    current_epoch, total_epochs, tqdm_able,
    lambda_local_alignment=0.0,
    lambda_global_alignment=0.0,
):
    """One training epoch.
    """
    net.train()
    sample_count = 0
    running_loss = 0.
    running_local_align_loss = 0.
    running_global_align_loss = 0.
    correct_count = 0

    with tqdm(
        train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False, unit="batch", disable=tqdm_able
    ) as pbar:
        for x, y, mask in pbar:
            # print(x.shape,y.shape)
            x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
            y_pred = net(x, mask)
            
            cls_loss = loss_fn(y_pred, y.to(torch.float32))
            aux_losses = getattr(_unwrap_model(net), "aux_losses", {})
            local_align_loss = aux_losses.get("local_align_loss", cls_loss.new_tensor(0.0))
            global_align_loss = aux_losses.get("global_align_loss", cls_loss.new_tensor(0.0))
            loss = (
                cls_loss
                + lambda_local_alignment * local_align_loss
                + lambda_global_alignment * global_align_loss
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]
            running_local_align_loss += local_align_loss.item() * x.shape[0]
            running_global_align_loss += global_align_loss.item() * x.shape[0]
            # binary classification with only one output neuron
            pred = (y_pred > 0.).int()
            correct_count += (pred == y).sum().item()

            pbar.set_postfix({
                "loss": running_loss / sample_count,
                "acc": correct_count / sample_count,
                "local": running_local_align_loss / sample_count,
                "global": running_global_align_loss / sample_count,
            })

    return {
        "loss": running_loss / sample_count,
        "acc": correct_count / sample_count,
        "local_align_loss": running_local_align_loss / sample_count,
        "global_align_loss": running_global_align_loss / sample_count,
    }


def val(
    net, val_loader, loss_fn, device, tqdm_able
):
    """Test the model on the validation / test set.
    """
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:
                # print(x.shape,y.shape)
                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                y_pred = net(x, mask)

                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                # binary classification with only one output neuron
                pred = (y_pred > 0.).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall) 
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": l, "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall) 
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (
        (TP + TN) / sample_count
        if sample_count > 0 else 0.0
    )
    return {
        "loss": l, "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
    }


def main():
    args = parse_args()

    gpu_ids = _parse_gpu_arg(args.gpu)
    if gpu_ids is None or not torch.cuda.is_available():
        primary_device = torch.device("cpu")
        dp_device_ids = None
    else:
        torch.cuda.set_device(gpu_ids[0])  # 设置默认 GPU
        primary_device = torch.device(f"cuda:{gpu_ids[0]}")
        dp_device_ids = gpu_ids if len(gpu_ids) > 1 else None

    print(f"[Device] primary={primary_device}, data_parallel_ids={dp_device_ids}")

    args.data_dir = os.path.join(args.data_dir,args.dataset)
    experiment_name = build_experiment_name(args)
    print(f"[Experiment] {experiment_name}")

    for i_iter in range(3):
        if args.if_wandb:
            wandb_run_name = f"{experiment_name}-{args.train_gender}-{args.test_gender}-iter{i_iter}"
            wandb.init(
                project="mamnba_ad", config=args, name=wandb_run_name,
            )
            args = wandb.config
        print(args)
        
        # Build Save Dir
        experiment_dir = os.path.join(args.save_dir, f"{experiment_name}_{i_iter}")
        samples_dir = os.path.join(experiment_dir, "samples")
        checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoints_dir, "best_model.pt")
        result_path = os.path.join("./results", f"{experiment_name}_{i_iter}.txt")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        print(f"[Paths] checkpoint={checkpoint_path}, result={result_path}")

        # construct the model
        if args.model == "DepMamba":
            if args.dataset=='lmvd':
                model_kwargs = dict(args.mmmamba_lmvd)
            elif args.dataset=='dvlog':
                model_kwargs = dict(args.mmmamba)
            model_kwargs.update({
                "use_local_alignment": args.use_local_alignment,
                "use_global_alignment": args.use_global_alignment,
            })
            net = DepMamba(**model_kwargs)# mmmamba_lmvd mmmamba
        else:#if args.model == "MAMBA":
            raise NotImplementedError(f"The {args.model} method has not been implemented by this repo")
        net = net.to(args.device[0])
        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)

        # prepare the data
        if args.dataset=='dvlog':
            train_loader = get_dvlog_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_dvlog_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_dvlog_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )
        elif args.dataset=='lmvd':
            train_loader = get_lmvd_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_lmvd_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_lmvd_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )

        # set other training components
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        best_val_acc = -1.0
        best_test_acc = -1.0
        if args.train:
            for epoch in range(args.epochs):
                train_results = train_epoch(
                    net, train_loader, loss_fn, optimizer, 
                    args.device[0], epoch, args.epochs, args.tqdm_able,
                    args.lambda_local_alignment,
                    args.lambda_global_alignment,
                )
                val_results = val(net, val_loader, loss_fn, args.device[0],args.tqdm_able)

                val_acc = (val_results["acc"] + val_results["precision"]+ val_results["recall"]+ val_results["f1"])/4.0
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(net.state_dict(), checkpoint_path)

                if args.if_wandb:
                    wandb.log({
                        "loss/train": train_results["loss"],
                        "acc/train": train_results["acc"],
                        "loss/local_align": train_results["local_align_loss"],
                        "loss/global_align": train_results["global_align_loss"],
                        "loss/val": val_results["loss"],
                        "acc/val": val_results["acc"],
                        "precision/val": val_results["precision"],
                        "recall/val": val_results["recall"],
                        "f1/val": val_results["f1"]
                    })
            
        # upload the best model to wandb website
        # load the best model for testing
        with torch.no_grad():
            net.load_state_dict(
                torch.load(checkpoint_path, map_location=args.device[0])
            )
            net.eval()
            test_results = val(net, test_loader, loss_fn, args.device[0],args.tqdm_able)
            print("Test results:")
            print(test_results)

            os.makedirs("./results", exist_ok=True)
            with open(result_path,'w') as f:    
                test_result_str = f'Accuracy:{test_results["acc"]}, Precision:{test_results["precision"]}, Recall:{test_results["recall"]}, F1:{test_results["f1"]}, Avg:{(test_results["acc"] + test_results["precision"]+ test_results["recall"]+ test_results["f1"])/4.0}'
                f.write(test_result_str)         

    if args.if_wandb:
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(checkpoint_path)
        wandb.run.summary["acc/best_val_acc"] = best_val_acc
        wandb.log_artifact(artifact)
        wandb.run.summary["acc/test_acc"] = test_results["acc"]
        wandb.run.summary["loss/test_loss"] = test_results["loss"]
        wandb.run.summary["precision/test_precision"] = test_results["precision"]
        wandb.run.summary["recall/test_recall"] = test_results["recall"]
        wandb.run.summary["f1/test_f1"] = test_results["f1"]

        wandb.finish()


if __name__ == '__main__':
    main()
