import gc
from typing import List, Iterable, Generator
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging


class E5LargeEmbedModel:
    def __init__(
        self,
        model_path: str = "e5_large",
        max_length: int = 512,
        prefix: str = "query: ",
        batch_size: int = None,
        logger_name: str = "logger",
    ):
        self.logger = logging.getLogger(logger_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.prefix = prefix
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.autocast_dtype = torch.float16 if self.device.type == "cuda" else None

        if self.batch_size is None:
            self._calibrate_batch_size()

    def get_batch_size(self):
        return self.batch_size

    def process_texts(
            self,
            texts: List[str]
    ) -> Generator[np.ndarray, None, None]:
        for embs in self.encode(texts):
            yield embs


    @staticmethod
    def _average_pool(
            last_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        summed = (last_hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def torch_cleanup(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def _calibrate_batch_size(self):
        if self.device.type != "cuda":
            self.batch_size = 1
            return
        text = Path("test_text.txt").read_text()

        def can_run(batch_size: int) -> bool:
            outputs = None
            embs = None
            batch_dict = None
            data = None

            try:
                data = [text] * batch_size
                batch_dict = self.tokenizer(
                    data,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

                if self.autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                        outputs = self.model(**batch_dict)
                        embs = E5LargeEmbedModel._average_pool(
                            outputs.last_hidden_state,
                            batch_dict["attention_mask"],
                        )
                        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                        embs = embs.detach().cpu().to(torch.float32).numpy()
                else:
                    outputs = self.model(**batch_dict)
                    embs = E5LargeEmbedModel._average_pool(
                        outputs.last_hidden_state,
                        batch_dict["attention_mask"],
                    )
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                    embs = embs.detach().cpu().to(torch.float32).numpy()

                _ = embs.shape
                torch.cuda.synchronize()
                return True

            except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
                return False

            except Exception as e:
                self.logger.exception(f"Unexpected error during batch calibration (1): {e}")
                raise

            finally:
                del outputs, embs, batch_dict, data
                self.torch_cleanup()

        batch_size = 8192

        while True:
            if batch_size < 1:
                self.logger.error("Even batch_size=1 does not fit into GPU memory")
                raise RuntimeError("Even batch_size=1 does not fit into GPU memory")

            self.logger.info(f"Batch size = {batch_size}")
            if can_run(batch_size):
                left_size = batch_size
                right_size = batch_size * 2
                break

            batch_size //= 2

        while right_size - left_size > 16:
            mid_size = (right_size + left_size) // 2
            self.logger.info(f"Batch size = {mid_size}")

            if can_run(mid_size):
                left_size = mid_size
            else:
                right_size = mid_size

        self.batch_size = max(1, int(left_size * 0.9))
        self.logger.info("Calibration done")

    @staticmethod
    def _batched(
            texts: List[str],
            batch_size: int
    ) -> Iterable[List[str]]:
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]

    @torch.inference_mode()
    def encode(
            self,
            texts: List[str]
    ) -> Generator[np.ndarray, None, None]:
        def get_embeds(_prepared):
            outputs = None
            embs = None
            batch_dict = None

            batch_dict = self.tokenizer(
                _prepared,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            try:
                if self.device.type == "cuda":
                    if self.autocast_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                            outputs = self.model(**batch_dict)
                            embs = E5LargeEmbedModel._average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                    else:
                        outputs = self.model(**batch_dict)
                        embs = E5LargeEmbedModel._average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                else:
                    outputs = self.model(**batch_dict)
                    embs = E5LargeEmbedModel._average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                embs = embs.detach().cpu().to(torch.float32).numpy()

            except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
                self.logger.error("Batch is too big")
                return None

            except Exception:
                self.logger.exception("Encoding failed")
                raise

            finally:
                del outputs, batch_dict
                self.torch_cleanup()

            return embs


        for i, sub_batch in enumerate(E5LargeEmbedModel._batched(texts, self.batch_size)):
            self.logger.info(f"Starting texts ({i * self.batch_size}-{(i + 1) * self.batch_size}) processing")
            prepared = [self.prefix + t for t in sub_batch]

            embs = get_embeds(prepared)
            if embs is None:
                prepared1 = prepared[:len(prepared) // 2]
                prepared2 = prepared[len(prepared) // 2:]
                embs1 = get_embeds(prepared1)
                embs2 = get_embeds(prepared2)
                embs = np.vstack((embs1, embs2))

            yield embs
