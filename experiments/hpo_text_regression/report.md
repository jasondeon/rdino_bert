# MentalBERT-only regression hyperparameter search

Objective: best-epoch, full-recording validation R-squared.
Classification is disabled, recording-level sampling uses four windows,
the LR scheduler is disabled, and all window choices use the same
30-second-eligible recording cohort.

| Trial | R2 | RMSE | Original RMSE | Best epoch | Parameters |
| ---: | ---: | ---: | ---: | ---: | --- |
| 36 | 0.325163 | 0.796615 | 8.737096 | 85 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 1, "learning_rate": 1.1962402885307259e-05, "lora_alpha": 8, "lora_rank": 8, "stride_seconds": 10.0, "weight_decay": 1e-06, "window_seconds": 25.0}` |
| 24 | 0.321241 | 0.798927 | 8.762453 | 85 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 1, "learning_rate": 1.083827506268681e-05, "lora_alpha": 8, "lora_rank": 8, "stride_seconds": 10.0, "weight_decay": 0.0001, "window_seconds": 25.0}` |
| 33 | 0.321173 | 0.798967 | 8.762889 | 85 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 1, "learning_rate": 1.074522623834033e-05, "lora_alpha": 8, "lora_rank": 8, "stride_seconds": 10.0, "weight_decay": 0.0001, "window_seconds": 25.0}` |
| 12 | 0.321141 | 0.798985 | 8.763094 | 130 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 1, "learning_rate": 8.590331178462853e-06, "lora_alpha": 8, "lora_rank": 8, "stride_seconds": 10.0, "weight_decay": 0.0001, "window_seconds": 25.0}` |
| 46 | 0.320976 | 0.799083 | 8.764160 | 85 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 2, "learning_rate": 1.3087267065053425e-05, "lora_alpha": 8, "lora_rank": 8, "stride_seconds": 10.0, "weight_decay": 0.0001, "window_seconds": 25.0}` |
| 32 | 0.319866 | 0.799736 | 8.771324 | 85 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 1, "learning_rate": 1.0630765041041398e-05, "lora_alpha": 8, "lora_rank": 8, "stride_seconds": 10.0, "weight_decay": 0.0001, "window_seconds": 25.0}` |
| 43 | 0.319566 | 0.799912 | 8.773259 | 85 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 1, "learning_rate": 1.0761957487332344e-05, "lora_alpha": 8, "lora_rank": 8, "stride_seconds": 10.0, "weight_decay": 0.0001, "window_seconds": 25.0}` |
| 34 | 0.319235 | 0.800106 | 8.775387 | 85 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 1, "learning_rate": 1.014717964113259e-05, "lora_alpha": 8, "lora_rank": 8, "stride_seconds": 10.0, "weight_decay": 0.0001, "window_seconds": 25.0}` |
| 4 | 0.318263 | 0.800677 | 8.781649 | 85 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 4, "learning_rate": 1.6358165301626213e-05, "lora_alpha": 16, "lora_rank": 4, "stride_seconds": 15.0, "weight_decay": 0.0001, "window_seconds": 20.0}` |
| 23 | 0.317910 | 0.800885 | 8.783927 | 85 | `{"disable_text_lora": false, "embedding_normalization": "batchnorm", "gradient_accumulation_steps": 1, "learning_rate": 1.0424707146088607e-05, "lora_alpha": 8, "lora_rank": 8, "stride_seconds": 10.0, "weight_decay": 0.0001, "window_seconds": 25.0}` |

## Parameter importance

- `disable_text_lora`: 0.5790
- `weight_decay`: 0.1573
- `learning_rate`: 0.1357
- `embedding_normalization`: 0.0807
- `gradient_accumulation_steps`: 0.0326
- `stride_seconds`: 0.0123
- `window_seconds`: 0.0025

LoRA rank and alpha are sampled only when LoRA is enabled.
Use the future untouched comparison set for the final model estimate;
the best value here is optimized on the current validation set.
