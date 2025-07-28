# Weight Loading Parallelism Guide

This guide summarizes how MBridge loads Hugging Face weights when multiple parallel strategies are enabled.

## ParallelStates
`ParallelStates` holds the size and rank of each parallel dimension (TP, PP, EP, etc.). It is usually constructed from Megatron-Core's parallel state and provides process groups for communication.

## Loading Weights
- **Tensor Parallel (TP) and Expert Tensor Parallel (ETP)**: only `tp_rank==0` or `etp_rank==0` reads the relevant weights from disk. After splitting with `_weight_split_across_tp`, the chunks are distributed to other ranks via `torch.distributed.scatter`.
- **Expert Parallel (EP)**: each expert rank loads its own experts. No communication is needed when reading weights.
- **Pipeline Parallel (PP/VPP)**: every pipeline stage invokes `load_weights` separately, loading only the parameters for its stage.
- **Special Case**: `lm_head.weight` is loaded on every TP rank when building a value model.

The core loop in `load_weights` shows the scatter logic:

```python
param_to_load = torch.empty_like(param)
if ".mlp.experts.linear_fc" in local_name:  # MoE/ETP weights
    if self.mpu.etp_rank == 0:
        mcore_weights_tp_split = self._weight_split_across_tp(
            local_name, mcore_weight, param, self.mpu.etp_size
        )
        mcore_weights_tp_split = [t.to(param.device) for t in mcore_weights_tp_split]
    else:
        mcore_weights_tp_split = None
    torch.distributed.scatter(
        param_to_load,
        mcore_weights_tp_split,
        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
        group=self.mpu.etp_group,
    )
else:  # regular weights via TP
    if self.mpu.tp_rank == 0:
        mcore_weights_tp_split = self._weight_split_across_tp(
            local_name, mcore_weight, param, self.mpu.tp_size
        )
        mcore_weights_tp_split = [t.to(param.device) for t in mcore_weights_tp_split]
    else:
        mcore_weights_tp_split = None
    torch.distributed.scatter(
        param_to_load,
        mcore_weights_tp_split,
        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
        group=self.mpu.tp_group,
    )
param.copy_(param_to_load)
```

(From `mbridge/core/bridge.py` lines 195–233.)

## Communication Methods
- `scatter` is used for TP and ETP to send the appropriate chunk to each rank.
- No `broadcast` occurs during weight loading. Broadcast utilities are used only in export/inference flows to share tensors or names across pipeline stages.

With this approach, disk I/O is minimized and each rank receives only the weights it needs.
