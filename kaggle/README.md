# Kaggle remote-GPU runner

Drives [`../llm_fine_tuning_LORA_task1_v2.ipynb`](../llm_fine_tuning_LORA_task1_v2.ipynb) on Kaggle's free
GPU as a batch kernel. Full rationale: [../docs/kaggle/option_a_kaggle_remote_gpu.md](../docs/kaggle/option_a_kaggle_remote_gpu.md).

## Before first run — replace the placeholder

Both metadata files contain `YOUR_KAGGLE_USERNAME`. Replace it with your Kaggle username in:

- `kernel-metadata.json` (`id` and `dataset_sources`)
- `dataset_payload/dataset-metadata.json` (`id`)

## One-time setup

> NOTE: on this machine the `kaggle.exe` shim is blocked by Windows Application Control,
> so always invoke the CLI as **`python -m kaggle`** (the venv already has it installed).

```powershell
.\env\Scripts\Activate.ps1        # use the project venv

# Auth: save the KGAT_... token (Account -> Settings -> API -> Create New Token) to access_token.
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Set-Content "$env:USERPROFILE\.kaggle\access_token" "KGAT_your_new_token" -NoNewline -Encoding ascii
python -m kaggle kernels list --mine        # auth check

# 1. Stage + upload the cleaned CSVs as a Kaggle Dataset
.\kaggle\stage_data.ps1
python -m kaggle datasets create -p kaggle\dataset_payload --dir-mode zip

# 2. First kernel push (creates the kernel)
.\kaggle\run.ps1

# 3. On kaggle.com -> your notebook -> Add-ons -> Secrets:
#    add a secret named exactly  HF_TOKEN  with your Hugging Face token, and enable it.
#    (Also accept the gated model license on huggingface.co with that same account.)
```

## Run loop

```powershell
.\kaggle\run.ps1 -Wait     # push, poll status, download adapter to ..\kaggle_output when complete
```

Later dataset updates:

```powershell
.\kaggle\stage_data.ps1
python -m kaggle datasets version -p kaggle\dataset_payload -m "update csvs" --dir-mode zip
```

The trained adapter (`adapter_model.safetensors`, `adapter_config.json`) lands under `..\kaggle_output\`.
