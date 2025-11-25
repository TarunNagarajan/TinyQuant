# Fisher Computation Notes

* Loss is rescaled to undo Hugging Face's mean reduction:
  `loss = outputs.loss * (inputs["input_ids"] != -100).sum().item()`

* Only valid (non-masked) tokens are counted; padding is ignored.

* CUDA cache clearing is tied to iteration index, not parameter count:
  `if i % 10 == 0: torch.cuda.empty_cache()`

* `.square()` replaces `.pow(2)` for gradients.

* `del inputs, outputs, loss` is used for consistent memory cleanup.

* Loop uses `enumerate()` to support scheduled cleanup.
