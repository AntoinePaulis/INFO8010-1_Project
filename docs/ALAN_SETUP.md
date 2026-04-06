# Alan Cluster Setup Guide

> **For:** Antoine Paulis  
> **Note:** You are on Windows — differences from the official README are noted explicitly.  
> Must be on **ULiège WiFi or VPN** for all Alan-related commands to work.

---

## 0. Prerequisites

- Request your own Alan account at [alan.montefiore.uliege.be/register](https://alan.montefiore.uliege.be/register) — accounts are **not shared**, each member needs one
- Save the private key file from your confirmation email somewhere accessible
- Have your Alan username from the confirmation email ready

---

## 1. SSH Setup (Windows-specific)

Use **PowerShell** or **Git Bash**. The SSH config file lives at `C:\Users\yourname\.ssh\config` instead of `~/.ssh/config` as shown in the README.

Create the `.ssh` folder if it doesn't exist:
```powershell
mkdir $HOME\.ssh
```

Copy the private key:
```powershell
copy path\to\privatekey $HOME\.ssh\id_alan
```

Create/edit `C:\Users\yourname\.ssh\config` and paste:
```
Host alan
  HostName master.alan.priv
  User your_alan_username
  IdentityFile ~/.ssh/id_alan
```

Test the connection:
```bash
ssh alan
```

> **If `ssh alan` fails but the full command works**, the config file wasn't saved correctly — check it exists at the right path. Retry creating it and make sure you save it without a `.txt` extension.

Full command fallback:
```bash
ssh -i ~/.ssh/id_alan your_alan_username@master.alan.priv
```

---

## 2. Conda Installation

Once connected to Alan via SSH:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh
sh Anaconda3-2023.07-1-Linux-x86_64.sh
```

When asked:
```
Do you wish the installer to initialize Anaconda3? [yes|no]
```
Answer **yes**.

After installation, reload the shell — conda won't be available until you do this:
```bash
source ~/.bashrc
```

---

## 3. Create the Conda Environment

```bash
conda create -n deep python=3.9 -c conda-forge
conda activate deep
```

---

## 4. Install PyTorch

> **Do NOT install Jax or TensorFlow** — only PyTorch is needed for this project.

The conda solver for PyTorch can hang indefinitely. Use pip instead:
```bash
pip install torch torchvision
```

---

## 5. Install Jupyter Kernel

`nb_conda_kernels` is not compatible with Python 3.9 via pip, and the conda solver hangs on it. Skip it and just run:

```bash
pip install ipykernel
python -m ipykernel install --user --name deep --display-name "deep"
```

This is enough to manually select the `deep` kernel in JupyterLab.

---

## 6. Create the Scratch Directory

The scratch directory does **not** exist by default — you must create it manually before any data transfer:

```bash
mkdir -p /scratch/users/your_alan_username
```

---

## 7. Dataset

The dataset is already on Andy's USB and has been transferred to his scratch. Ask him to copy it to your scratch from Alan:

```bash
cp -r /scratch/users/andyjalloh/Dataset /scratch/users/your_alan_username/
```

---

## 8. Clone the Repo

```bash
cd /home/your_alan_username
git clone https://github.com/yourrepo
```

---

## 9. Set PROJECT_ENV

Add the environment variable to your `.bashrc` so it's set automatically on every login:

```bash
echo 'export PROJECT_ENV=alan' >> ~/.bashrc
source ~/.bashrc
```

---

## 10. Day-to-day Workflow

| Task | Where |
|------|--------|
| Write code | Locally in VSCode |
| Push changes | `git push` from local |
| Pull changes on Alan | `git pull` from `/home/your_alan_username/yourreponame` |
| Run preprocessing | `python code/src/preprocess.py` on Alan terminal |
| Submit training jobs | `sbatch code/jobs/train_ball.sh` on Alan terminal |
| Monitor training | [wandb.ai](https://wandb.ai) from anywhere |
| Explore data / debug | JupyterLab on Alan via browser |

> Must be on **ULiège WiFi or VPN** for SSH and JupyterLab access.

---

## 11. JupyterLab Access

Go to [https://alan.montefiore.uliege.be/jupyter](https://alan.montefiore.uliege.be/jupyter) and log in with your Alan credentials.

To switch from JupyterNotebook to JupyterLab, change `tree` to `lab` in the URL:
```
https://alan.montefiore.uliege.be/jupyter/user/you/lab
```

**Connect VSCode directly to Alan's Jupyter (recommended):**
1. Install the **Jupyter** and **Python** extensions in VSCode
2. Open a notebook
3. Click the kernel selector (top right) → **Existing Jupyter Server**
4. Enter `https://alan.montefiore.uliege.be/jupyter`
5. Log in and select the `deep` kernel

**Reconnection**

```bash
ssh alan
conda activate deep
cd INFO8010-1_Project/
```

Go to the dataset

```bash
ls /scratch/users/andyjalloh/
```