"""Opt-in storage bounds for the validated eager Step5 PCP deployment."""
import os


def compact_dsa_layers(config):
    from vllm_hcu.v1.step5_pcp_config import is_step5_pcp

    flag = os.environ.get('VLLM_STEP5_PCP_COMPACT_DSA_STATE', '0')
    if flag not in ('0', '1'):
        raise ValueError('VLLM_STEP5_PCP_COMPACT_DSA_STATE must be 0 or 1')
    enabled = flag == '1' and is_step5_pcp(config)
    if enabled and not config.model_config.enforce_eager:
        raise ValueError('Step5 PCP local-layer DSA storage requires eager')
    return enabled


def indexer_rope_limit(config):
    """Keep speculative lookahead, without changing frequencies or scaling."""
    from vllm_hcu.v1.step5_pcp_config import is_step5_pcp

    flag = os.environ.get('VLLM_STEP5_PCP_TRIM_INDEXER_ROPE', '0')
    if flag not in ('0', '1'):
        raise ValueError('VLLM_STEP5_PCP_TRIM_INDEXER_ROPE must be 0 or 1')
    if flag == '0' or not is_step5_pcp(config):
        return 0
    if not config.model_config.enforce_eager:
        raise ValueError('Step5 PCP indexer RoPE storage trimming requires eager')
    maximum = int(config.model_config.max_model_len)
    if maximum <= 0:
        raise ValueError('Positive serving context length required')
    spec = getattr(config, 'speculative_config', None)
    lookahead = int(getattr(spec, 'num_speculative_tokens', 0))
    if lookahead not in (0, 1, 2, 3):
        raise ValueError('Only the validated Step5 MTP lookahead is supported')
    # Include invalid speculative rows near the end of the context. The
    # cache builder uses length only for arange, never to set frequencies.
    return ((maximum + lookahead + 1 + 63) // 64) * 64


def rotary_cache_position(config, original_maximum, parameters):
    """Bound ordinary rotary tables for the verified eager PCP model only."""
    name = 'VLLM_STEP5_PCP_TRIM_ROTARY_CACHE'
    flag = os.environ.get(name, '0')
    if flag not in ('0', '1'):
        raise ValueError(name + ' must be 0 or 1')
    if flag == '0':
        return original_maximum
    from vllm_hcu.v1.step5_pcp_config import is_step5_pcp
    if not is_step5_pcp(config):
        return original_maximum
    if not config.model_config.enforce_eager:
        raise ValueError('Step5 PCP rotary storage trimming requires eager')
    if parameters.get('rope_type', 'default') not in ('default', 'llama3') or parameters.get('factor', 1.) != 1.:
        raise ValueError('Only prefix-verified default/llama3 factor1 rotary tables are supported')
    maximum = int(config.model_config.max_model_len)
    lookahead = int(getattr(getattr(config, 'speculative_config', None), 'num_speculative_tokens', 0))
    if maximum <= 0 or lookahead not in (0, 1, 2, 3):
        raise ValueError('Invalid serving position or unverified MTP lookahead')
    bound = ((maximum + lookahead + 1 + 63) // 64) * 64
    return min(int(original_maximum), bound)
