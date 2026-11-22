# SPDX-License-Identifier: Apache-2.0
"""Step5 speculative decode input preparation for the HCU V2 runner."""

import numpy as np
import torch
import vllm.envs as envs
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
)


class Step5DecodeInputs:
    def __init__(self):
        self.key = None
        self.entry = None

    def _indices(
        self, slots: np.ndarray, logits: np.ndarray, queries: np.ndarray,
        query_buffer: torch.Tensor, total_logits: int, decode_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compare exact values. Equal batch sizes need not have equal slot
        # order, draft lengths, or padded query offsets. Keep streams isolated.
        stream = torch.cuda.current_stream(query_buffer.device)
        key = (stream, slots.tobytes(), logits.tobytes(), queries.tobytes(),
               total_logits, decode_width)
        if key != self.key or self.entry is None:
            idx_words = 2 * slots.size  # int64 slots in units of int32
            q_offset = idx_words + logits.size
            host = torch.empty(
                q_offset + queries.size, dtype=torch.int32,
                device="cpu", pin_memory=True,
            )
            words = host.numpy()
            words[:idx_words].view(np.int64)[:] = slots
            words[idx_words:q_offset] = logits
            words[q_offset:] = queries
            packed = host.to(query_buffer.device, non_blocking=True)
            idx = packed[:idx_words].view(torch.int64)
            cu = packed[idx_words:q_offset]
            expanded, local = expand_idx_mapping(idx, total_logits, cu, decode_width)
            self.entry = (packed[q_offset:], (idx, cu, expanded, local))
            self.key = key

        query_values, indices = self.entry
        # Dummy or other uncached steps can overwrite the graph-owned buffer.
        # Restore its full padded contents even on a hit, at the original address.
        query_buffer.copy_(query_values, non_blocking=True)
        return indices

    def prepare_inputs(self, runner, scheduler_output, batch_req_state, batch_desc):
        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
        assert num_tokens > 0
        if envs.VLLM_MOE_SKIP_PADDING:
            # Mark trailing cudagraph-padding rows so kernels can skip work for
            # them when supported.
            is_padding = runner.input_buffers.is_padding
            is_padding[:num_tokens].fill_(False)
            is_padding[num_tokens:num_tokens_after_padding].fill_(True)

        req_ids = batch_req_state.req_ids
        num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        num_reqs = len(req_ids)
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_bonus_tokens = runner.model_state.num_new_sampled_tokens_per_step
        total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
        total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
        num_logits = num_draft_tokens_per_req + num_bonus_tokens
        # combine_sampled_and_draft_tokens places a request's logits rows
        # at [query_end - num_logits, query_end). Fewer query rows than
        # that would silently select the preceding request's hidden states.
        assert (num_scheduled_tokens_np >= num_logits).all()
        cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
        cu_num_logits_np[0] = 0
        np.cumsum(num_logits, out=cu_num_logits_np[1:])

        # Get query_start_loc.
        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(runner.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        query_start_loc = runner.input_buffers.query_start_loc
        idx_mapping, cu_num_logits, expanded_idx_mapping, expanded_local_pos = self._indices(
            idx_mapping_np, cu_num_logits_np, query_start_loc_np, query_start_loc,
            total_num_logits, runner.decode_query_len,
        )
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = query_start_loc[: num_reqs_padded + 1]
        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            runner.req_states.num_computed_tokens.gpu,
            runner.input_buffers.positions,
            runner.input_buffers.seq_lens,
        )
        seq_lens = runner.input_buffers.seq_lens[:num_reqs_padded]

        dcp_local_seq_lens = None
        if runner.use_dcp:
            # Prepare dcp local seq_lens.
            prepare_dcp_local_seq_lens(
                runner.input_buffers.dcp_local_seq_lens,
                runner.input_buffers.seq_lens,
                num_reqs,
                runner.dcp_size,
                runner.dcp_rank,
                runner.cp_interleave,
            )
            dcp_local_seq_lens = runner.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            runner.input_buffers.input_ids,
            idx_mapping,
            runner.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            runner.req_states.prefill_len.gpu,
            runner.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            runner.model_state.num_new_sampled_tokens_per_step,
        )

        # CPU upper bound on seq_lens; padded entries left at zero.
        num_computed_tokens_np = runner.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens_np,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        prompt_lens = None
        if runner.model_config.rswa_window is not None:
            # prompt_lens is only used in R-SWA case.
            prompt_lens = runner.req_states.prompt_len.gpu[idx_mapping]

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens_np,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            input_ids=runner.input_buffers.input_ids[:num_tokens_after_padding],
            positions=runner.input_buffers.positions[:num_tokens_after_padding],
            is_padding=runner.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=prompt_lens,
            max_query_len=None,
        )
