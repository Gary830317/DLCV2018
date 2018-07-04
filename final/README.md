# DLCV Final Challenge 1

# Task 2 

## Output:
```
bash task2_output.sh <path/to/novel> <path/to/test> <output/path>
```
- 例如: bash task2_output.sh task2_dataset/novel/ test/ ./

- 會吐出 10_shot.csv, 1_shot.csv, 5_shot.csv 分別為 10-shots, 5-shots, 1-shot 的結果。

## Train:
```
bash task2_train.sh <#shots> <path/to/task2_dataset/base> <path/to/task2_dataset/novel> <output/path>
```
- 例如: bash task2_train.sh 10 task2_dataset/base/ task2_dataset/novel/ ./

- 會在 output path 存該 shot 的 model weight。

- 因為 train 的過程當中包含使用 base images 對 Resnet 的 training，跑一次大約會花上一天的時間。這裡我們也提供我們已經在 base data 上 train 好的 model，可以大幅降低 training的時間。 助教只需到 train.py 這個檔中將 use_trained_resnet 更改為 True 及可。
