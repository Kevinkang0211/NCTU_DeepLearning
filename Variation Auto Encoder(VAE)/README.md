
[VAE_report.pdf](https://github.com/Kevinkang0211/NCTU_DeepLearning/files/6498325/VAE_report.pdf)

![image](https://user-images.githubusercontent.com/45477381/118601987-aa265f00-b7e4-11eb-9e58-cb8c3aeb2abf.png)

![image](https://user-images.githubusercontent.com/45477381/118602019-b14d6d00-b7e4-11eb-906f-27794111058e.png)

![image](https://user-images.githubusercontent.com/45477381/118602091-d0e49580-b7e4-11eb-88e8-d873b961133d.png)

![image](https://user-images.githubusercontent.com/45477381/118602108-d8a43a00-b7e4-11eb-8e75-c4458f0779e1.png)

## VAE 問題討論:
經過實驗結果下來，Kullback-Leiblier (KL) term 乘上不同的 λ 值，會使訓
練結果有好壞的差別。
當 λ 值越小，train 過程的 Loss 可以降得較低，出來的 VAE 模型可以有更
好的還原效果；反之，λ 值越大，train 過程的 Loss 可以會比較高，出來
的 VAE 模型還原效果較差
