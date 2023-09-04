import torch
import torch.utils.data
import time
from socket import gethostname
import torch.distributed as dist
from pathlib import Path


def itr_restart(it):
    while True:
        for x in it:
            yield x


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def experiment_info(args, device, model, output, len_dataset, len_iterator):
    print(args, file=output)
    print("Number of GPU devices: ", torch.cuda.device_count(), file=output)
    print("Device used: ", device, file=output)
    print("Model information: ", model, file=output)
    print('# training with Adam: lr={}, weight_decay={}'.format(args.lr, args.weight_decay), file=output)
    print(f"Length dataset: {len_dataset}, length dataset iterator: {len_iterator}", file=output)


def multi_gpu_info(rank, world_size, gpus_per_node, local_rank):
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}, world_size: {world_size}"
          f", gpus_per_node: {gpus_per_node}")
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)


def multitask_train_info(writer, loss, acc, i, data="train", name="SIMILARITY"):
    writer.add_scalars(name + '/loss', {data: loss}, i)
    writer.add_scalars(name + '/acc', {data: acc}, i)


def save_checkpoint(model, optimizer, step, experiment_id, use_multi_gpu, scheduler):
    print("\n****Saving checkpoint*****")

    save_path = f"/scratch/ii1g17/protein-embeddings/saved_models/{experiment_id}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_path = f"{save_path}/iter_{step}_checkpoint.pt"

    checkpoint = {
        'model': model.module.state_dict() if use_multi_gpu else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, save_path)


def save_model(experiment_id, use_multi_gpu, model, i):
    print("\n****Saving model*****")

    save_path = f"/scratch/ii1g17/protein-embeddings/saved_models/{experiment_id}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_path = f"{save_path}/iter_{i}.pt"
    torch.save(model.module.state_dict(), save_path) if use_multi_gpu else torch.save(model.state_dict(), save_path)
