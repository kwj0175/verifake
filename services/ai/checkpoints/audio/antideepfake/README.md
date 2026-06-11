# AntiDeepfake weights

Place AntiDeepfake model weights in this directory.

## Default file names

- `mms_300m.ckpt` - default checkpoint used by `services/ai/audio_pipeline/antideepfake.py`

## Default execution paths

- Wrapper entrypoint: `python -m services.ai.audio_pipeline.antideepfake <audio-file>`
- Vendored entrypoint: `python services/ai/antideepfake/main.py inference services/ai/antideepfake/hparams/mms_300m_audio_pipeline.yaml --base_path <repo-root> --ckpt_path services/ai/checkpoints/audio/antideepfake/mms_300m.ckpt --test_csv <protocol.csv>`

## Notes

- Do not commit large checkpoint files to git.
- If you keep a different filename, pass it explicitly with `--checkpoint-path` in the wrapper or `--ckpt_path` in the vendored script.
- TODO: add more checkpoint naming guidance if additional upstream models are enabled in this repo.
