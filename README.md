# MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation

<p align="center">
    <a href="">
        <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-XXXX">
    </a>
    <a href="">
        <img alt="Build" src="https://img.shields.io/badge/🤗">
    </a>
</p>

<p align="center">
    <img src="assets/task_example.png" alt="Task Example" width="800">
</p>

## Dataset

The data structure at the {huggingface} is as follows.

```
data/
├── 3_back/
│   ├── 006_0.jpg
│   ├── 006_1.jpg
│   ├── 006_2.jpg
│   ├── 006_prompt.txt
│   ├── 014_0.jpg
│   ├── 014_1.jpg
│   ├── 014_2.jpg
│   ├── 014_prompt.txt
│   └── ...
├── 3_global/
│   └── ...
├── 3_local/
│   └── ...
└── ...
```


## Setup

```bash
git clone git@github.com:matsuolab/multibanana.git
cd multibanana

conda create -n multibanana python=3.12

pip install -r requirements.txt
```


## Evaluation

Generated images are expected to be saved in the same directory with the `_generated` suffix.

```
data/
├── 3_back/
│   ├── 006_0.jpg
│   ├── 006_1.jpg
│   ├── 006_2.jpg
│   ├── 006_prompt.txt
│   ├── 006_generated.jpg
│   ├── 014_0.jpg
│   ├── 014_1.jpg
│   ├── 014_2.jpg
│   ├── 014_prompt.txt
│   ├── 014_generated.jpg
│   └── ...
├── 3_global/
│   └── ...
├── 3_local/
│   └── ...
└── ...
```

We use `gemini-2.5`-flash via the Google GenAI SDK, and `gpt-5` via the OpenAI SDK.

Please set your API key in `.env` as follows

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...

```

Run

```bash
# Gemini
python judge.py --base_dir ./data --model gemini --batch_size 32 --output_dir ./results

# GPT
python judge.py --base_dir ./data --model gpt --batch_size 32 --output_dir ./results
```

This will evaluate all generated images and save the results in `{number}_{model}_judge.txt` files (e.g., `006_gemini_judge.txt`).

## License

Apache-2.0 license

## Citation

```bibtex
@inproceedings{oshima2025multibanana,
  title={MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation},
  author={Yuta Oshima and Daiki Miyake and Kohsei Matsutani and Yusuke Iwasawa and Masahiro Suzuki and Yutaka Matsuo and Hiroki Furuta},
  year={2025}
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={},
}
```
