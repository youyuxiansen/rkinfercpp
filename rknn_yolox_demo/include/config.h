//
// Created by Felix youyuxiansen@gmail.com on 2021/12/8.
//

//#define YOLOV5S_ORIGIN
// for model best.rknn
#define YOLOV5S_INDOOR

#ifdef YOLOV5S_INDOOR
#define LABEL_NAME_TXT_PATH "./model/amicro_indoor_labels_list.txt"
// 标签中类的数目
#define OBJ_CLASS_NUM     16
// 三个输出Detect对应使用的anchor的大小
#define ANCHOR0 {59, 40, 83, 106, 171, 57}
#define ANCHOR1 {143, 157, 112, 316, 206, 227}
#define ANCHOR2 {383, 194, 248, 361, 436, 362}
#endif

// for model yolov5s_relu_rv1109_rv1126_out_opt.rknn
#ifdef YOLOV5S_ORIGIN
#define LABEL_NAME_TXT_PATH "./model/coco_80_labels_list.txt"
#define OBJ_CLASS_NUM     80
#define ANCHOR0 {10, 13, 16, 30, 33, 23}
#define ANCHOR1 {30, 61, 62, 45, 59, 119}
#define ANCHOR2 {116, 90, 156, 198, 373, 326}
#endif