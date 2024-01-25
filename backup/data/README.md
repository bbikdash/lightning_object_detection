# 





## Training on a custom dataset

We've designed this functionality to be similar to how (UltraLytics)[https://github.com/ultralytics/ultralytics] and (Alexey Bochkovskiy)[https://github.com/AlexeyAB/darknet] allow training of YOLO with custom datasets. The [YoloV5 wiki contains a good example](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

Datasets are organized to have images and labels. Labels must be in the _YOLO format_ with one `*.txt` file per image (if no objects in image, no `*.txt` file is required). The `*.txt` file specifications are:
 * One row per object
 * Each row is `class x_center y_center width height` format.
 * Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.
 * Class numbers are zero-indexed (start from 0).

