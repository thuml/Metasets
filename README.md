## MetaSets: Meta Learning on Point Sets for Generalizable Representations

### Datasets

We use ModelNet, ShapeNet and ScanObjectNN as our training and test dataset. In benchmark ModelNet -> ScanObjectNN, we use 11 categories. In benchmark ShapeNet -> ScanObjectNN, we use 9 categories.  Download links are shown below.

* [ModelNet_11](https://www.dropbox.com/s/87rtldo5dxg2q76/modelnet_11.zip?dl=0)
* [ScanObjectNN_11](https://www.dropbox.com/s/kdz2g95yj8kjlfs/scanobjectnn_11.zip?dl=0)
* [ShapeNet_9](https://www.dropbox.com/s/n3rbqv2hlb51csm/shapenet_9.zip?dl=0)
* [ScanObjectNN_9](https://www.dropbox.com/s/z0y5c1v9skjcxsb/scanobjectnn_9.zip?dl=0)

Download all these datasets and put them into dir `Data`. The file tree will be like this:

```
|--data
   |--modelnet_11
   |--scanobjectnn_11
   |--shapenet_9
   |--scanobjectnn_9
```

### Pretrained Model

We trained model on task ModelNet -> ScanObjectNN and ShapeNet -> ScanObjectNN. Download links are shown below.

* [ModelNet -> ScanObjectNN](https://www.dropbox.com/s/gjl13uqjhoskpx6/best_model_model2scan.pt?dl=0)
* [ShapeNet -> ScanObjectNN](https://www.dropbox.com/s/xxuyec1verdnz0t/best_model_shape2scan.pt?dl=0)

Download all these models and put them into dir `models`. The file tree will be like this:

```
|--models
   |--best_model_model2scan.pt
   |--best_model_shape2scan.pt
```

After download all models and datasets, and put them into dirs, the whole file tree will be like this:

```
|--data
|--models
|--data_utils.py
|--dataloader.py
|--model_pointnet_meta.py
|--model_utils_meta.py
|--test.py
|--train.py
```

### Requirements

```python
torch==1.4.0
torchvision==0.5.0
torchmeta
h5py
matplotlib
numpy
Cuda 10.0
Nvidia-driver >=440
```

 ### Train

#### Modelnet -> ScanObjectNN

```bash
# assemble that 4 gpus are available
python train.py -source modelnet_11 -target scanobjectnn_11 -g 0,1,2,3
```

#### ShapeNet -> ScanObjectNN

```bash
python train.py -source shapenet_9 -target scanobjectnn_9 -g 0,1,2,3
```

### Evaluate

#### Modelnet -> ScanObjectNN

```bash
python test.py ./data/scanobjectnn_11 models/best_model_model2scan.pt
```

#### ShapeNet -> ScanObjectNN

```bash
python test.py ./data/scanobjectnn_9 models/best_model_shape2scan.pt
```

#### Results

* Benchmark I: ModelNet -> ScanObjectNN 68.63%
* Benchmark II: ShapeNet -> ScanObjectNN 56.75%

