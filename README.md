# README.md

## 結果

トレーニングプロセスの結果は `results` ディレクトリにまとめられています。これには、`train.py` の実行による出力が含まれます。

# results ディレクトリの命名規則について
- `black`: 白背景に黒文字を意味します。
- `white`: 黒背景に白文字を意味します。
- `m_black+r_both`: このディレクトリは、MNISTデータセットの画像は `black` 、ランダムパターンは差分を取っていることを示します。
- `m_both+r_both`: このディレクトリは、どちらも差分を取っていることを示します。

## ディレクトリ構成

```
/img_rec/
├── README.md
├── data
│   ├── mnist
│   └── random
├── notebooks
│   ├── demo
│   └── exploration
├── results/pix28/
│   ├── m_black+r_both/gidc/...
│   └── m_both+r_both/gidc/...
└── src
    ├── models/ ... モデルを.pyで定義
    ├── utils/  ... ユーティリティを定義
    ├── config.py ... pathを定義
    ├── trainer.py ... ループを定義
    └── train.py ... 実行ファイル
```

- img_recディレクトリから`python -m src.train`で実行