

## 使用OK-VQA数据集中的other类别对LLaVA进行微调

微调数据格式:

```json
[
    {
        "id": "unique_id",
        "image": "image_file.jpg",
        "conversations": [
            {
    
                "from": "human",
                "value": "What is shown in the image?"
    
            },
            {
                "from": "gpt",
                "value": "formatted_answers"
            }
        ]
    }
 
]
```



## OK-VQA: A Visual Question Answering Benchmark CVPR 2019

- 作者因为标准VQA数据集质量不高（难度低），所以自行请MTurk工人，从COCO数据集中进行了数据采集、问题搜集、问题质量筛选、问题回答。同时通过过滤操作，降低了bias的影响，减少文本对于某些回答的偏差（如 Is there ...）。同时考虑了长尾效应。
- 就数据分类而言，划分了10+1（other）个类别，保证问题类型的互斥。

![img](https://raw.gitmirror.com/da5sdasddasa/image/main/202403221050676.png)



### Train

DeepSpeed 是一个开源深度学习优化库，旨在提高大规模深度学习模型训练的速度、规模和效率。它由 Microsoft 开发，通过利用各种优化技术，允许更快、更高效的训练，特别是对于非常大的模型。

DeepSpeed 的关键组件之一是其 ZeRO 技术。ZeRO 旨在优化训练期间的内存使用，从而能够训练比以前在相同硬件上训练更大的模型。ZeRO 分为不同的优化阶段，ZeRO 阶段 2 就是其中之一。ZeRO Stage 2 通过在数据并行进程中对优化器状态、梯度和参数进行分区来减少内存冗余。这意味着每个进程仅存储这些组件的一部分，从而大大降低了每个进程的内存需求。如果您在使用此配置时遇到 CUDA 内存错误，请考虑尝试第 3 阶段配置，该配置允许将梯度卸载到 CPU，这会减慢训练速度，但可能会解决内存错误。

finetune_script:

```shell
#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./llava-v1.5-7b \
    --version v1 \
    --data_path ./ok_vqa_dataset/train/dataset.json \
    --image_folder ./ok_vqa_dataset/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-ok_vqa-lora_bs16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

```



gpu_mem:

![image-20240322114438345](https://raw.gitmirror.com/da5sdasddasa/image/main/202403221144463.png)



WandB:

![image-20240322181642922](https://raw.gitmirror.com/da5sdasddasa/image/main/202403221816022.png)



### Inference

http://172.31.119.115:7860/

微调前：

![image-20240322141229677](https://raw.gitmirror.com/da5sdasddasa/image/main/202403221412780.png)

微调后：

![image-20240322141247290](https://raw.gitmirror.com/da5sdasddasa/image/main/202403221412378.png)