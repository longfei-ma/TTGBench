import os
import torch
import torch.nn as nn
from torch.utils.data import Sampler, RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.trainer import has_length
from typing import Dict, Optional, Sequence


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"): #False
        return unwrap_model(model.module)
    else:
        return model
    
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    # to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    to_return = {k: v.detach().cpu() for k, v in to_return.items()}
    return to_return


class GraphChatTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset): #False
            return None

        if self.args.shuffle: 
            return RandomSampler(self.train_dataset)
        else:
             return SequentialSampler(self.train_dataset)

    # def _save(self, output_dir: Optional[str] = None, state_dict=None):
    #     if getattr(self.args, 'tune_graph_mlp_adapter', False): # True
    #         # Save the model
    #         _state_dict = state_dict
    #         if _state_dict is None: #True
    #             # Only save the model itself if we are using distributed training
    #             model_to_save = unwrap_model(self.model)
    #             _state_dict = model_to_save.state_dict()

    #         weight_to_save = {}
    #         keys_to_match = ['graph_projector', 'embed_tokens', 'embed_in']
    #         for k, v in _state_dict.items():
    #             if any(key_match in k for key_match in keys_to_match):
    #                 weight_to_save[k] = v

    #         current_folder = output_dir.split('/')[-1]
    #         parent_folder = os.path.dirname(output_dir)
    #         if current_folder.startswith('checkpoint-'): #True
    #             mm_projector_folder = os.path.join(parent_folder, "graph_projector")
    #             os.makedirs(mm_projector_folder, exist_ok=True)
    #             torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
    #         else:
    #             torch.save(weight_to_save, os.path.join(output_dir, f'graph_projector.bin'))
    #     else:
    #         super(GraphChatTrainer, self)._save(output_dir, state_dict)#在checkpoint-2目录及输出目录下保存了配置文件和模型参数pytorch_model-00001-of-00003.bin等文件

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_graph_mlp_adapter', False): # True
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            keys_to_match = ['graph_projector', 'embed_tokens', 'embed_in'] #实际只保存了前两个

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            self.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, f'graph_projector.bin'))
        else:
            super(GraphChatTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_graph_mlp_adapter', False): # True
            pass
        else:
            super(GraphChatTrainer, self)._save(output_dir, state_dict)