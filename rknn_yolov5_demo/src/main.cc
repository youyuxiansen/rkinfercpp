// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>

#define _BASETSD_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "drm_func.h"
#include "rga_func.h"
#include "rknn_api.h"
#include "postprocess.h"
#include "VideoProcessor.h"
#include "utils.h"
#include "media.h"

#define PERF_WITH_POST 1

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
           "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2],
           attr->dims[1], attr->dims[0], attr->n_elems, attr->size, 0, attr->type,
           attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

int doInference(cv::Mat &img, void *drm_buf, int img_width,
                int img_height, int channel, int model_width, int model_height,
                rga_context &rga_ctx, rknn_input *inputs, rknn_context &ctx,
                rknn_input_output_num &io_num, rknn_tensor_attr *output_attrs,
                std::vector<float> &out_scales, std::vector<uint8_t> &out_zps,
                rknn_output *outputs)
{
    int ret;
    void *resize_buf = malloc(model_height * model_width * channel);

    img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, model_width, model_height);
    // img_resize_fast(&rga_ctx, drm_buf, img_width, img_height, resize_buf, width, height);
    inputs[0].buf = resize_buf;
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    if (resize_buf)
    {
        free(resize_buf);
    }
    return ret;
}

void detect(cv::Mat &img, void *drm_buf, int img_width,
            int img_height, int channel, int model_width, int model_height,
            rga_context &rga_ctx, rknn_input *inputs, rknn_context &ctx,
            rknn_input_output_num &io_num, rknn_tensor_attr *output_attrs,
            std::vector<float> &out_scales, std::vector<uint8_t> &out_zps,
            rknn_output *outputs, float conf_threshold, float nms_threshold, float vis_threshold,
            detect_result_group_t &detect_result_group)
{
    doInference(img, drm_buf, img_width, img_height,
                channel, model_width, model_height, rga_ctx,
                inputs, ctx, io_num, output_attrs, out_scales, out_zps,
                outputs);
    // post process
    float scale_w = (float)model_width / img_width;
    float scale_h = (float)model_height / img_height;
    post_process((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, model_height,
                 model_width,
                 conf_threshold, nms_threshold, vis_threshold, scale_w, scale_h, out_zps, out_scales,
                 &detect_result_group);
    rknn_outputs_release(ctx, io_num.n_output, outputs);
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
    rknn_context ctx;
    int drm_fd;
    int buf_fd = -1; // converted from buffer handle
    unsigned int handle;
    size_t actual_size = 0;
    int img_width;
    int img_height;
    rga_context rga_ctx;
    drm_context drm_ctx;
    const float vis_threshold = 0.1;
    const float nms_threshold = 0.5;
    const float conf_threshold = 0.3;
    //    const float nms_threshold = 0.45;
    //    const float conf_threshold = 0.25;
    struct timeval start_time, stop_time;
    int ret;
    memset(&rga_ctx, 0, sizeof(rga_context));
    memset(&drm_ctx, 0, sizeof(drm_context));

    if (argc != 4)
    {
        printf("Usage: %s <rknn model> <1(img)/2(vdo)/3(dir)> <jpg/mp4/avi/dir> \n", argv[0]);
        return -1;
    }

    char *model_name = argv[1];
    int source_type = atoi(argv[2]);
    char *source = argv[3];
    std::string path(source);
    int ps = path.find_last_of("/");
    std::string filename_with_suffix = path.substr(ps + 1);

    printf("create_media...source_type=%d\n", source_type);
    BaseMedia *media = create_media(source_type, path);

    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        printRKNNTensor(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        width = input_attrs[0].dims[0];
        height = input_attrs[0].dims[1];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
    }

    printf("model input height=%d, width=%d, channel=%d\n", height, width,
           channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    rknn_output outputs[io_num.n_output];

    // DRM alloc buffer
    drm_fd = drm_init(&drm_ctx);
    // init rga context
    RGA_init(&rga_ctx);

    cv::Mat orig_img;
    while (1)
    {
        ret = media->get_media(orig_img);
        if (ret)
        {
            std::cout << "get_media ended and returns " << ret << std::endl;
            return ret;
        }
        std::cout << "orig_img.cols:" << orig_img.cols << "; orig_img.rows" << orig_img.rows << std::endl;
        img_width = orig_img.cols;
        img_height = orig_img.rows;
        gettimeofday(&start_time, NULL);
        std::vector<float> out_scales;
        std::vector<uint8_t> out_zps;
        detect_result_group_t detect_result_group;
        void *drm_buf = NULL;
        drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, img_width, img_height,
                                channel * 8, &buf_fd, &handle, &actual_size);
        memcpy(drm_buf, orig_img.data, img_width * img_height * channel);
        detect(orig_img, drm_buf, img_width,
               img_height, channel, width, height,
               rga_ctx, inputs, ctx,
               io_num, output_attrs,
               out_scales, out_zps,
               outputs, conf_threshold, nms_threshold, vis_threshold,
               detect_result_group);
        drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
        gettimeofday(&stop_time, NULL);
        printf("once run use %f ms\n",
               (__get_us(stop_time) - __get_us(start_time)) / 1000);

        // Draw Objects and save under model/detected_
        media->write_detected(orig_img, detect_result_group);
        // plot_bbox_on_img(orig_img, detect_result_group);

        // video.write(orig_img);
    }
    // release
    ret = rknn_destroy(ctx);

    drm_deinit(&drm_ctx, drm_fd);
    RGA_deinit(&rga_ctx);
    if (model_data)
    {
        free(model_data);
    }

    return 0;
}
