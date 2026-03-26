# FATHOM CONSTANT — Cross-Architecture Universality Test

**Date:** 2026-03-26 06:41:04
**Duration:** 898.1s
**Pairs:** 30 (paper power replication set, p=0.000051)

## Hypothesis

If the surface/insight depth ratio converges across different transformer
architectures (within ±0.2), the Fathom Constant is a universal property
of how transformers separate retrieval from reasoning.

## Architecture Compatibility

circuit-tracer requires **transcoders** (not SAEs) for each model architecture.
Currently supported transcoder sets:

| Transcoder Set | Models | Source |
|---------------|--------|--------|
| `gemma` | google/gemma-2-2b | mwhanna/gemma-scope-transcoders |
| `llama` | meta-llama/Llama-3.2-1B | mntss/transcoder-Llama-3.2-1B |

**Not supported:** GPT-2 (gpt2, gpt2-xl). The package `jbloom/GPT2-Small-SAEs-Reformatted`
provides SAEs, not transcoders. These are incompatible with circuit-tracer's
`ReplacementModel.from_pretrained()` which requires transcoders with `config.yaml`.

## Models Tested

| Model | Transcoder | Pairs | Surface Mean | Insight Mean | Delta | Ratio | % Expected |
|-------|-----------|-------|-------------|-------------|-------|-------|-----------|
| google/gemma-2-2b | gemma | 30 | 8.2076 | 7.9668 | +0.2408 | 1.0302 | 76.7% |
| meta-llama/Llama-3.2-1B | llama | ERROR | — | — | — | — | — |

## Result

| Metric | Value |
|--------|-------|
| **Candidate Fathom Constant (weighted mean)** | **1.0302** |
| Models successfully tested | 1/2 |
| Cross-architecture validation | **PENDING** (single model) |
| Total pairs scored | 30 |

## Interpretation

**CANDIDATE VALUE: 1.0302** (single model: google/gemma-2-2b). Cross-architecture validation pending — need ≥2 successful models to confirm universality. Cannot claim convergence with one data point.

## Per-Model Ratios

- **google/gemma-2-2b**: 1.0302

## Errors

### meta-llama/Llama-3.2-1B

**Error:** You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-1B.
403 Client Error. (Request ID: Root=1-69c510c3-18034c7107af61df08d40d39;dfbae35e-7cb6-4641-9996-02580eaad243)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json.
Your request to access model meta-llama/Llama-3.2-1B is awaiting a review from the repo authors.

```
Traceback (most recent call last):
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\utils\_http.py", line 403, in hf_raise_for_status
    response.raise_for_status()
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\requests\models.py", line 1026, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\utils\hub.py", line 479, in cached_files
    hf_hub_download(
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\utils\_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py", line 1014, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py", line 1121, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py", line 1662, in _raise_on_head_call_error
    raise head_call_error
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py", line 1550, in _get_metadata_or_catch_error
    metadata = get_hf_file_metadata(
               ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\utils\_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py", line 1467, in get_hf_file_metadata
    r = _request_wrapper(
        ^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py", line 283, in _request_wrapper
    response = _request_wrapper(
               ^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py", line 307, in _request_wrapper
    hf_raise_for_status(response)
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\utils\_http.py", line 420, in hf_raise_for_status
    raise _format(GatedRepoError, message, response) from e
huggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: Root=1-69c510c3-18034c7107af61df08d40d39;dfbae35e-7cb6-4641-9996-02580eaad243)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json.
Your request to access model meta-llama/Llama-3.2-1B is awaiting a review from the repo authors.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\heyzo\clawd\sae-reasoning-depth\fathom_constant_final.py", line 347, in test_model
    scorer = DepthScorer(
             ^^^^^^^^^^^^
  File "C:\Users\heyzo\clawd\sae-reasoning-depth\depth_scorer.py", line 206, in __init__
    self._load_model(model_name, model_type, dtype)
  File "C:\Users\heyzo\clawd\sae-reasoning-depth\depth_scorer.py", line 218, in _load_model
    self._model = ReplacementModel.from_pretrained(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\clawd\research\circuit-tracer\circuit_tracer\replacement_model\replacement_model.py", line 62, in from_pretrained
    return cls.from_pretrained_and_transcoders(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\clawd\research\circuit-tracer\circuit_tracer\replacement_model\replacement_model.py", line 114, in from_pretrained_and_transcoders
    return TransformerLensReplacementModel.from_pretrained_and_transcoders(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\clawd\research\circuit-tracer\circuit_tracer\replacement_model\replacement_model_transformerlens.py", line 111, in from_pretrained_and_transcoders
    model = super().from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformer_lens\HookedTransformer.py", line 1376, in from_pretrained
    state_dict = loading.get_pretrained_state_dict(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformer_lens\loading_from_pretrained.py", line 2344, in get_pretrained_state_dict
    hf_model = AutoModelForCausalLM.from_pretrained(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\models\auto\auto_factory.py", line 549, in from_pretrained
    config, kwargs = AutoConfig.from_pretrained(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\models\auto\configuration_auto.py", line 1332, in from_pretrained
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\configuration_utils.py", line 662, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\configuration_utils.py", line 721, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\utils\hub.py", line 322, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\heyzo\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\utils\hub.py", line 543, in cached_files
    raise OSError(
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-1B.
403 Client Error. (Request ID: Root=1-69c510c3-18034c7107af61df08d40d39;dfbae35e-7cb6-4641-9996-02580eaad243)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json.
Your request to access model meta-llama/Llama-3.2-1B is awaiting a review from the repo authors.

```

