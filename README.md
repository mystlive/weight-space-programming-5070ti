# RTX 5070 Tiで重みプログラミングを追試

GPT-6 AstraがQwen3.5-0.8Bの重みを直接編集したExperiment 8を、公開差分から再構成して追試しました。 [研究記事](https://rikiyat.github.io/blog/weight-space-programming/)
重みファイルのSHA-256は公開修正版と一致しました。 [照合記録](models/provenance.json)

## 実測結果

MATH-500から選んだ同じ12問を4条件で試しました。

| モデル | 考えるモード | 正解 | 上限到達 | 出力トークン数の中央値 |
| --- | --- | ---: | ---: | ---: |
| 元版 | ON | 3/12 | 9 | 32,768 |
| 元版 | OFF | 6/12 | 0 | 954.5 |
| 修正版 | ON | 7/12 | 3 | 10,405 |
| 修正版 | OFF | 7/12 | 0 | 3,315.5 |

48件の生成に合計3時間54分41秒かかりました。
生成中のPyTorchのVRAM予約量は最大5,494 MiBでした。

## 計測ファイル

- [数学8問の計測](runs/math8-32768)では難度1と2を各4問試しました。
- [追加4問の計測](runs/math4-32768)では難度3を2問、難度4と5を各1問試しました。
- [試走](runs/pilot-4096-static)では数学、日本語、指示追従、繰り返しを誘発する課題を各8問試しました。

各ディレクトリの `config.json` に実行条件、`summary.json` に集計、条件別のJSONに出力全文、正誤、出力トークン数、推論時間、停止理由を保存しています。
`length` は出力上限への到達、`eos` は終了トークンの生成です。
推論時間の定義と計測環境は各 `config.json` に記載しています。

## 再構成モデル

[Release v1.0.0](https://github.com/mystlive/weight-space-programming-5070ti/releases/tag/v1.0.0) からBF16の再構成モデル一式を入手できます。
ZIPを展開し、中の `qwen3.5-0.8b-p50c-rebuilt` をTransformersのモデルディレクトリとして指定します。
同梱の `weights.sha256` で重みファイルを照合できます。

## 再実行

計測ではWindows 11、Python 3.12.10、RTX 5070 Ti、PyTorch 2.13.0+cu130、Transformers 5.20.2を使用しました。
下記のコマンドはリポジトリのルートでPowerShellから実行します。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.resolved.txt
.\.venv\Scripts\python.exe -B -c 'import prepare; prepare.prepare_models()'

$env:PYTORCH_ALLOC_CONF = 'roundup_power2_divisions:4,garbage_collection_threshold:0.75'
.\.venv\Scripts\python.exe -B run.py --run-id repro-math8 --category math --batch-size 4 --max-new-tokens 32768
.\.venv\Scripts\python.exe -B run.py --run-id repro-math4 --category math --batch-size 4 --max-new-tokens 32768 --prompts runs/math4-32768/prompts.json
```

同梱のプロンプトを使用し、再実行の出力は `runs/repro-math8` と `runs/repro-math4` に保存されます。
公開用の `run.py` には入力ファイルを指定する `--prompts` を追加しています。
本計測で使ったソースは[当時の `run.py`](https://github.com/mystlive/weight-space-programming-5070ti/blob/237dc9411c2e68732d49a366467ca7fd9d329e82/run.py)です。

## 出典とライセンス

- 元モデルは[Qwen3.5-0.8Bの固定版](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)です。モデルのApache 2.0ライセンスはモデルZIPに同梱しています。
- 編集データと適用コードはRikiya Takehi、Jayden Teoh、Alex Zhang、Omar Khattabの[公開ログ](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math)に由来します。 [上流ライセンス](upstream/LICENSE)
- 数学の出題は[MATH-500の固定版](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be)から抽出しました。元の[MATHのライセンス](licenses/MATH-MIT.txt)と[PRM800Kのライセンス](licenses/PRM800K-MIT.txt)を同梱しています。
