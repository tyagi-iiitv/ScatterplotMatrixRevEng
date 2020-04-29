# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3 for scatterplot matrix reverse engineering, with support for training, inference and evaluation.

## Installation (Colab)
### Clone
```
!git clone https://github.com/siddhantrele/ScatterplotMatrixRevEng  
%cd ScatterplotMatrixRevEng/
```

## Train on Custom Dataset

#### Custom model
Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset.

If you want to change num_classes (default is 3, points, ticks, labels), delete existing config/yolov3-custom.cfg before running below commands.
```
%cd config/                       # Navigate to config dir                         
!sh create_custom_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg'
```

#### Classes
Add/Change class names in `data/custom/classes.names`. This file should have one row per class name. Default is already added, points, ticks and labels

Add/Change num_classes in `config/custom.data`

#### Image Folder
To generate training and testing images:
```
!python generate_scatterplots_train_test.py
```
Images will be generated in `data/custom/images/` with corresponding labels in `data/custom/labels/` and `data/custom/train.txt` and `data/custom/valid.txt` will be populated with the paths to the generated images.

#### Annotation Description
Each row in the annotation file defines one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates are scaled `[0, 1]`, and the `label_idx` is zero-indexed and corresponds to the row number of the class name in `data/custom/classes.names`.

#### Train
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]

```

To train on the custom dataset run:

```
!pip uninstall tensorflow                   # Uninstall TF 2.0
!pip install tensorflow==1.13.2             # Install compatible TF version
!pip install -r requirements.txt            # Install required modules
!python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
```

#### Training log
```
---- [Epoch 7/100, Batch 7300/14658] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
+------------+--------------+--------------+--------------+
| grid_size  | 16           | 32           | 64           |
| loss       | 1.554926     | 1.446884     | 1.427585     |
| x          | 0.028157     | 0.044483     | 0.051159     |
| y          | 0.040524     | 0.035687     | 0.046307     |
| w          | 0.078980     | 0.066310     | 0.027984     |
| h          | 0.133414     | 0.094540     | 0.037121     |
| conf       | 1.234448     | 1.165665     | 1.223495     |
| cls        | 0.039402     | 0.040198     | 0.041520     |
| cls_acc    | 44.44%       | 43.59%       | 32.50%       |
| recall50   | 0.361111     | 0.384615     | 0.300000     |
| recall75   | 0.222222     | 0.282051     | 0.300000     |
| precision  | 0.520000     | 0.300000     | 0.070175     |
| conf_obj   | 0.599058     | 0.622685     | 0.651472     |
| conf_noobj | 0.003778     | 0.004039     | 0.004044     |
+------------+--------------+--------------+--------------+
Total Loss 4.429395
---- ETA 0:35:48.821929
```

#### Draw bounding boxes
weights_path should point to the checkpoint of the trained model you want to use, all images inside image_folder will have bounding box detection executed on them, and output will be in the "outputs" folder.
```
!python detect.py \
--image_folder data/custom/images \
--model_def config/yolov3-custom.cfg \
--class_path data/custom/classes.names \
--weights_path checkpoints/yolov3_ckpt_90.pth
```

## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
