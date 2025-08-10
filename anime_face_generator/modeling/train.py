import io
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from anime_face_generator.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()





def train_model(   model_config,
                   train_config,
                   train_loader,
                   val_loader,
                   device='cpu',
                   T=1000,
                   BATCH_SIZE=64,
                   beta=None,
                   alpha=None,
                   alpha_bar=None,
                   get_loss=None,
                   optimizer=None,
                   model_class=None,
                   wandb=True):
    torch.cuda.empty_cache()
    model = model_class(model_config)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if device.type == 'cuda':
        model = model.cuda()

    # os.environ["NCCL_DEBUG"]="INFO"
    # os.environ["WANDB_NOTEBOOK_NAME"] = "1.60-VDM-Giga-UNet-parallelized-training.ipynb"

    # Training Loop
    DATA_SIZE = len(train_loader.dataset)
    epoch = 1   # start epoch
    global_step = 0

    if wandb:
        import wandb
        # Start a new wandb run to track this script.
        run = wandb.init(
            name=f'Parallel-GIGA-DDPM[{time.strftime("%Y-%m-%d_%H-%M-%S")}]',
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="ayoub-dev2000-university-of-liege",
            # Set the wandb project where this run will be logged.
            project="Diffusion model - Anime face Generator",
            # Track hyperparameters and run metadata.
            config=model_config | train_config,
            
        #     mode="offline",
        )

    max_loss = 100
    while epoch < train_config['num_epochs']:
        print("---------------------------------------------")
        print(" EPOCH " + str(epoch))
        print("---------------------------------------------")
        
        for batch, (x0_batch, _) in enumerate(train_loader):
            model.train()
            x0_batch = x0_batch.cuda(non_blocking=True) if device == 'cuda' else x0_batch.to(device)
            t = torch.randint(0, T, (x0_batch.shape[0],), device=x0_batch.device).long()

            train_loss, pred_noise, x_t = get_loss(model, x0_batch, t)
            
            if train_loss.item() > max_loss:
                print("Loss exploded! Loss value: ", train_loss.item())
            
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run.log({"train_loss": train_loss.item(), "epoch": epoch, "train_batch": batch}, step=global_step)
            
            if batch % (800) == 0:
            #############   INTERMIDIATE EVALUATION     ##################
                model.eval()
                with torch.no_grad():
                    mean_val_loss = []
                    for i, (x0_batch_val, _) in enumerate(val_loader):   # get a batch (large enough) from the validation set at random
                        x0_batch_val = x0_batch_val.cuda(non_blocking=True) if device.type == 'cuda' else x0_batch_val.to(device)
                        t = torch.randint(0, T, (x0_batch_val.shape[0],), device=x0_batch_val.device).long()
        
                        val_loss, pred_noise, x_t = get_loss(model, x0_batch_val, t)
                        mean_val_loss.append(val_loss.item())
                        if i > 150:
                            break
                
                    val_loss_mean = np.mean(mean_val_loss)
                    run.log({"val_loss": val_loss_mean, "epoch": epoch, "val_batch": batch}, step=global_step)
                
                current = batch * BATCH_SIZE + len(x0_batch)
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

                beta_t = beta[t_0].view(-1, 1, 1, 1)
                sqrt_alpha_t = alpha[t_0].sqrt().view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_bar_t = (1 - alpha_bar[t_0]).sqrt().view(-1, 1, 1, 1)

                # Compute predicted x_0
                pred_x0 = (xt - sqrt_one_minus_alpha_bar_t.detach().cpu() * eps_theta) / sqrt_alpha_t.detach().cpu()
                plt.subplot(1, 4, 4)
                plt.imshow(np.clip(((pred_x0[0].permute(1, 2, 0) + 1) / 2), 0, 1))
                plt.title("Denoised original")
                plt.axis("off")

                plt.tight_layout()
                
                wandb.log({"evolution":  f"{current:>5d}/{DATA_SIZE:>5d}", "train_loss": train_loss, "mean_val_loss": val_loss_mean}, step=global_step)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                wandb.log({"Denoising a sample from validation set": wandb.Image(Image.open(buf))}, step=global_step)
                plt.show()
                plt.close()
            global_step += 1
            

            
        print(f"Epoch {epoch} finished")
        # checkpoint
        if epoch % 1 == 0:
            ckpt_name = f"85M_params_GIGA_DDPM_Unet_ckpt_epoch_{epoch}.pth"
            path = MODELS_DIR / ckpt_name
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), path)
            else:
                torch.save(model.state_dict(), path)
            print(f"Model saved as {ckpt_name}")

        epoch += 1
            

    run.finish()
    print("finished")        




@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
