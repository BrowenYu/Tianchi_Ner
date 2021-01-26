# 中药NER


## Prepare

### Environment
* Ubuntu 16.04, CUDA 9.0, GCC 4.9.4
* Python 3.6.6

* Tensorflow 1.12.0 Keras 2.2.4 

* bert4keras0.8.8  

 

### Data
#### 解压 ./data 下的round1_train.zip和round1_test.zip文件
* 将训练数据放在./data/round1_train 下
* 将测试数据放在./data/round1_test/chusai_xuanshou 下

### Run

python3.6 data_process_cv.py 

python3.6 main.py

生成的提交文件在./data/round1_test/submission里

复赛加入了flat和知识蒸馏方法，代码分别在distillation.py 和train_flat.py

