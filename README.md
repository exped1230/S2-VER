# S2-VER
The official implement of paper S2-VER: Semi-Supervised Visual Emotion Recognition.

To perform S2-VER on your dataset with 1600 labels, run:

```python main.py --epoch 512 --num_train_iter 1024 --num_labels 1600 -bsz 8 --train_data_dir your_train_set_path --test_data_dir your_test_set_path -ds fi -nc 8 --num_workers 4 --gpu 1 --overwrite```

The used datasets are provided in our [homepage](http://47.105.62.179:8081/sentiment/index.html)
