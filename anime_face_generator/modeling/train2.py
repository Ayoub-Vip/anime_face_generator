import io
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from anime_face_generator.config import MODELS_DIR, PROCESSED_DATA_DIR
import torch.utils.data as tdata

from pathlib import Path

from loguru import logger
from tqdm import tqdm
# import typer
from dataclasses import asdict


def train_model(   module,
                   dataset,
                   device='cpu',
                   rand_gen=torch.Generator().manual_seed(42),
                   model_backup=None,
                   epoch_start=1,
                   step_start=0,
                   wandb_logs=True):
    
    train_config = module.TrainConfig()
    model_config = module.ModelParams()
    scheduler_config = module.SchedulerConfig()
    model = module.NoisePredictor()
    scheduler = module.LinearScheduleDiffuser() #if train_config.scheduler_type == 'linear' else module.CosineScheduler()
    if model_backup is not None:
        model.load_state_dict(torch.load(MODELS_DIR / model_backup, weights_only=True))

    train_dataset, val_dataset = tdata.random_split(
        dataset, [0.9, 0.1],
        generator=rand_gen
    )

    train_loader = tdata.DataLoader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )

    val_loader = tdata.DataLoader(
        dataset=val_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )

    torch.cuda.empty_cache()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if device.type == 'cuda':
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    
    # os.environ["NCCL_DEBUG"]="INFO"
    # os.environ["WANDB_NOTEBOOK_NAME"] = "1.60-VDM-module-UNet-parallelized-training.ipynb"

    # Training Loop
    DATA_SIZE = len(train_loader.dataset)
    epoch = epoch_start   # start epoch
    global_step = step_start

    if wandb_logs:
        import wandb
        # Start a new wandb run to track this script.
        run = wandb.init(
            name=f'{model_config.model_name}[{time.strftime("%Y-%m-%d_%H-%M-%S")}]',
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="ayoub-dev2000-university-of-liege",
            # Set the wandb project where this run will be logged.
            project="Diffusion model - Anime face Generator",
            # Track hyperparameters and run metadata.
            config= asdict(model_config) | asdict(scheduler_config) | asdict(train_config),
            # Track the code in this file.
            save_code=True,
            
            # mode="offline",
        )

    max_loss = 100
    while epoch < train_config.num_epochs:
        print("---------------------------------------------")
        print(" EPOCH " + str(epoch))
        print("---------------------------------------------")
        
        for batch, (x0_batch, _) in enumerate(train_loader):
            model.train()
            x0_batch = x0_batch.cuda(non_blocking=True) if device == 'cuda' else x0_batch.to(device)
            t = torch.randint(0, module.SchedulerConfig.T, (x0_batch.shape[0],), device=x0_batch.device).long()

            train_loss, pred_noise, x_t = model.module.get_loss(x0_batch, t)
            
            if train_loss.item() > max_loss:
                print("Loss exploded! Loss value: ", train_loss.item())
            
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if wandb_logs:
                run.log({"train_loss": train_loss.item(), "epoch": epoch, "train_batch": batch}, step=global_step)
            
            if batch % (800) == 0:
            #############   INTERMIDIATE EVALUATION     ##################
                model.eval()
                with torch.no_grad():
                    mean_val_loss = []
                    for i, (x0_batch_val, _) in enumerate(val_loader):   # get a batch (large enough) from the validation set at random
                        x0_batch_val = x0_batch_val.cuda(non_blocking=True) if device.type == 'cuda' else x0_batch_val.to(device)
                        t = torch.randint(0, module.SchedulerConfig.T, (x0_batch_val.shape[0],), device=x0_batch_val.device).long()
        
                        val_loss, pred_noise, x_t = model.module.get_loss(x0_batch_val, t)
                        mean_val_loss.append(val_loss.item())
                        if i > 150:
                            break
                
                    val_loss_mean = np.mean(mean_val_loss)
                    if wandb_logs:
                        run.log({"val_loss": val_loss_mean, "epoch": epoch, "val_batch": batch}, step=global_step)
                
                current = batch * train_config.batch_size + len(x0_batch)
                print(f"train loss: {train_loss.item():>7f}  [{current:>5d}/{DATA_SIZE:>5d}]")
                print(f"val loss: {val_loss_mean:>7f}  [{current:>5d}/{DATA_SIZE:>5d}]")
                
                print("timestep in first x in the batch: ", t[0].item())
                
                # Visualize one sample from the validataion batch
                idx = 0
                img = x0_batch_val[idx].detach().cpu()
                eps_theta = pred_noise[idx].detach().cpu()
                xt = x_t[idx].detach().cpu()
                t_0 = t[idx].detach().cpu()
        
                plt.figure(figsize=(8, 2))
                plt.subplot(1, 4, 1)
                plt.imshow(np.clip((img.permute(1, 2, 0) + 1) / 2, 0, 1))
                plt.title("original input")
                plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.imshow(np.clip((xt.permute(1, 2, 0) + 1) / 2, 0, 1))
                plt.title("noised original t=" + str(t[0].item()))
                plt.axis("off")
                
                plt.subplot(1, 4, 3)
                plt.imshow(np.clip((eps_theta.permute(1, 2, 0) + 1) / 2, 0, 1))
                plt.title("predicted noise")
                plt.axis("off")

                beta_t = scheduler.get_beta_t(t_0)
                sqrt_alpha_t = (1 - beta_t).sqrt().view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_bar_t = (1 - scheduler.get_alpha_bar_t(t_0)).sqrt().view(-1, 1, 1, 1)

                # Compute predicted x_0
                pred_x0 = (xt - sqrt_one_minus_alpha_bar_t.detach().cpu() * eps_theta) / sqrt_alpha_t.detach().cpu()
                plt.subplot(1, 4, 4)
                plt.imshow(np.clip(((pred_x0[0].permute(1, 2, 0) + 1) / 2), 0, 1))
                plt.title("Denoised original")
                plt.axis("off")

                plt.tight_layout()
                if wandb_logs:
                    wandb.log({"evolution":  f"{current:>5d}/{DATA_SIZE:>5d}", "train_loss": train_loss, "mean_val_loss": val_loss_mean}, step=global_step)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                if wandb_logs:
                    wandb.log({"Denoising a sample from validation set": wandb.Image(Image.open(buf))}, step=global_step)
                plt.show()
                plt.close()
            global_step += 1
            

            
        print(f"Epoch {epoch} finished")
        # checkpoint
        if epoch % 1 == 0:
            ckpt_name = f"{model_config.model_name}_ckpt_epoch_{epoch}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth"
            path = MODELS_DIR / ckpt_name
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), path)
            else:
                torch.save(model.state_dict(), path)
            print(f"Model saved as {ckpt_name}")

        epoch += 1
            
    if wandb_logs:
        run.finish()
    print("finished")
    return model     



# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Training some model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Modeling training complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
