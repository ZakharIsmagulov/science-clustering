from Config import PeripheryConfig
from HdbscanModel import HdbscanModel
import cupy as cp


class PeripheryModel:
    def __init__(self, config: PeripheryConfig):
        self.core_sample_frac = config.core_sample_frac
        self.core_sample_min = config.core_sample_min
        self.core_sample_max = config.core_sample_max
        self.core_quantile = config.core_quantile
        self.mv_batch_size = config.mv_batch_size
        self.batch_size = config.batch_size

        self.tau = None
        self.delta = None

    def calibrate_thresholds_from_core(
            self,
            hdbscan_model: HdbscanModel,
            cp_reduced: cp.array,
            seed: int | None = None,
    ):
        core_idx = cp.where(hdbscan_model.get_labels() != -1)[0]
        if core_idx.size == 0 and (self.tau is None or self.delta is None):
            raise ValueError("There is no core sample")

        core_size = int(core_idx.size)
        sample_size = int(core_size * self.core_sample_frac)
        sample_size = max(self.core_sample_min, sample_size)
        sample_size = min(self.core_sample_max, sample_size, core_size)

        rng = cp.random.RandomState(seed)
        pick = rng.choice(core_idx.size, size=sample_size, replace=False)
        core_idx_sample = core_idx[pick]

        mempool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()

        p1_all = cp.empty(sample_size, dtype=cp.float32)
        margin_all = cp.empty(sample_size, dtype=cp.float32)
        write_pos = 0
        for start in range(0, sample_size, self.batch_size):
            end = min(start + self.batch_size, sample_size)
            idx_b = core_idx_sample[start:end]
            X_b = cp_reduced[idx_b]

            mv_b = hdbscan_model.get_cupy_membership_probs(X_b, self.mv_batch_size)

            p1_b = mv_b.max(axis=1)

            top2_b = cp.partition(mv_b, kth=mv_b.shape[1] - 2, axis=1)[:, -2:]
            p2_b = cp.minimum(top2_b[:, 0], top2_b[:, 1])
            margin_b = p1_b - p2_b

            done_count = end - start
            p1_all[write_pos:write_pos + done_count] = p1_b.astype(cp.float16, copy=False)
            margin_all[write_pos:write_pos + done_count] = margin_b.astype(cp.float16, copy=False)
            write_pos += done_count

            del X_b, mv_b, top2_b, p1_b, p2_b, margin_b
            cp.cuda.Device().synchronize()
            mempool.free_all_blocks()
            pinned_pool.free_all_blocks()

        self.tau = float(cp.quantile(p1_all, self.core_quantile))
        self.delta = float(cp.quantile(margin_all, self.core_quantile))
        cp.cuda.Device().synchronize()
        mempool.free_all_blocks()
        pinned_pool.free_all_blocks()

    def attach_noise_by_membership(
            self,
            hdbscan_model: HdbscanModel,
            cp_reduced: cp.array,
    ):
        noise_idx = cp.where(hdbscan_model.get_labels() == -1)[0]
        if noise_idx.size == 0:
            return

        mempool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()

        new_labels = hdbscan_model.get_labels()
        final_probs = hdbscan_model.get_probs()
        n_noise = int(noise_idx.size)
        for start in range(0, n_noise, self.batch_size):
            end = min(start + self.batch_size, n_noise)
            idx_b = noise_idx[start:end]
            X_b = cp_reduced[idx_b]

            mv_b = hdbscan_model.get_cupy_membership_probs(X_b, self.mv_batch_size)

            c1_b = mv_b.argmax(axis=1)
            p1_b = mv_b.max(axis=1)

            top2_b = cp.partition(mv_b, kth=mv_b.shape[1] - 2, axis=1)[:, -2:]
            p2_b = cp.minimum(top2_b[:, 0], top2_b[:, 1])

            margin_b = p1_b - p2_b
            ok_b = (p1_b >= self.tau) & (margin_b >= self.delta)

            if bool(ok_b.any()):
                idx_ok = idx_b[ok_b]
                new_labels[idx_ok] = c1_b[ok_b].astype(new_labels.dtype, copy=False)
                final_probs[idx_ok] = p1_b[ok_b].astype(cp.float16, copy=False)

            del X_b, mv_b, top2_b, c1_b, p1_b, p2_b, margin_b, ok_b
            cp.cuda.Device().synchronize()
            mempool.free_all_blocks()
            pinned_pool.free_all_blocks()
