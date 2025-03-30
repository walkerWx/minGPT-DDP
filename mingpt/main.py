from torch.distributed import init_process_group, destroy_process_group
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from trainer import Trainer, TrainerConfig
from char_dataset import CharDataset, DataConfig
import torch
import os
import hydra
from omegaconf import DictConfig

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK")))

def ddp_cleanup():
    destroy_process_group()



def get_train_objs(gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size

    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)

    return model, optimizer, train_set, test_set

@hydra.main(config_path=".", config_name="gpt2_train_cfg.yaml")
def main(cfg: DictConfig):
    ddp_setup()

    gpt_cfg = GPTConfig(**cfg['gpt_config'])
    opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
    data_cfg = DataConfig(**cfg['data_config'])
    trainer_cfg = TrainerConfig(**cfg['trainer_config'])

    model, optimizer, train_set, test_set = get_train_objs(gpt_cfg, opt_cfg, data_cfg)
    trainer = Trainer(trainer_cfg, model, optimizer, train_set, test_set)

    trainer.train()

    destroy_process_group()
    
if __name__ == "__main__":
    main()
