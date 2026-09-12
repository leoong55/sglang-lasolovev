"""Bounded GPU draft windows with a CPU backing store for radix/HiCache.

The target keeps its original virtual IDs. Only the draft attention input uses
request-local ring slots. All committed context K/V is retained on CPU, so an
L1 prefix hit, L2 relocation or a new shared-prefix branch can refill its window.
This implementation favors explicit completion over asynchronous host races;
its CPU transfers and window-miss checks must be measured on hardware.
"""

import logging
import threading
import weakref

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.glm53_draft_layout import bounded_draft_geometry
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)


class GLM53BoundedDraftPool(MHATokenToKVPool):
    is_glm53_bounded = True

    def __init__(self, logical_size, *, max_requests, **kwargs):
        page = kwargs["page_size"]
        if kwargs.get("post_capture_active", False):
            raise ValueError("Bounded draft does not support post-capture pool resizing")
        if kwargs.get("enable_memory_saver", False):
            raise ValueError("Bounded draft does not support memory-saver offload")
        kwargs["kv_cache_layout"] = "nhd"
        if (kwargs["head_num"], kwargs["head_dim"], kwargs["v_head_dim"],
                kwargs["layer_num"], kwargs["dtype"]) != (1, 128, 128, 6, torch.bfloat16):
            raise ValueError("Bounded GLM53 draft requires six BF16 layers, one TP-local KV head and head_dim128")
        rows, self.ring_stride, self.fixed_gpu_bytes = bounded_draft_geometry(max_requests, page)
        super().__init__(rows, **kwargs)
        self.logical_size = int(logical_size)
        self.max_requests = int(max_requests)
        self.window = 2048
        self.backing_lock = threading.RLock()
        # Token-major permits one D2H per append, then one CPU indexed scatter.
        self.backing = torch.empty(
            (logical_size + page, 6, 2, 1, 128), dtype=self.dtype,
            device="cpu", pin_memory=str(self.device).startswith("cuda"),
        )
        self.backing_valid = torch.zeros(logical_size + page, dtype=torch.bool)
        self.logical_versions = torch.zeros(logical_size + page, dtype=torch.int32, device=self.device)
        self.slot_virtual = torch.full((rows + page,), -1, dtype=torch.int64, device=self.device)
        self.slot_version = torch.zeros(rows + page, dtype=torch.int32, device=self.device)
        self.restore_epoch = 0
        self.seen_restore_epoch = 0
        self.fastpath_enabled = envs.SGLANG_GLM53_BOUNDED_DRAFT_FASTPATH.get()
        # Only request identity and CPU metadata are retained. Weak references
        # must not extend a finished request's (or its prefix tensor's) lifetime.
        self._window_owners = {}
        self._continuity_ready = False
        self.window_checks = 0
        self.window_reuses = 0
        for buffer in (*self.k_buffer, *self.v_buffer):
            buffer.zero_()
        logger.info(
            "GLM53 bounded draft: window=2048, requests=%d, GPU rows=%d (+page%d), "
            "GPU fixed=%.3f GiB, logical rows=%d, CPU backing=%.3f GiB; HiCache uses CPU-backed virtual IDs",
            max_requests, rows, page, self.fixed_gpu_bytes / 2**30,
            logical_size, self.backing.numel() * self.backing.element_size() / 2**30,
        )
        logger.info("GLM53 bounded draft continuity fast path: %s", self.fastpath_enabled)

    def invalidate_continuity(self):
        self._window_owners.clear()
        self._continuity_ready = False

    @staticmethod
    def _owner_state(owner):
        """Snapshot scheduler metadata without reading a GPU tensor.

        Repointing a radix prefix replaces prefix_indices. Retraction/readmission
        also changes these fields and passes through prefill, which explicitly
        invalidates all owners. Never infer identity from a client request ID.
        """
        kv = getattr(owner, "kv", None)
        prefix = getattr(owner, "prefix_indices", None)
        slot = getattr(kv, "req_pool_idx", None)
        if slot is None or not isinstance(prefix, torch.Tensor):
            return None
        state = (
            int(slot), getattr(kv, "cache_protected_len", None),
            bool(getattr(owner, "is_retracted", False)),
            bool(getattr(owner, "retracted_stain", False)),
        )
        if state[2]:
            return None
        try:
            return weakref.ref(owner), weakref.ref(prefix), state
        except TypeError:
            # Unknown/custom request objects keep the validated legacy path.
            return None

    def _can_reuse_window(self, owners, batch_size):
        if (not self.fastpath_enabled or not self._continuity_ready
                or owners is None or len(owners) != batch_size):
            return False
        for owner in owners:
            current = self._owner_state(owner)
            if current is None:
                return False
            previous = self._window_owners.get(current[2][0])
            if (previous is None or previous[0]() is not owner
                    or previous[1]() is not current[1]()
                    or previous[2] != current[2]):
                return False
        return True

    def _remember_window_owners(self, owners, batch_size):
        if owners is None or len(owners) != batch_size:
            self.invalidate_continuity()
            return
        for owner in owners:
            state = self._owner_state(owner)
            if state is not None:
                self._window_owners[state[2][0]] = state

    def fill_seq_lens_cpu_bound(self, *, prefix_lens_cpu, reserved_lens_cpu,
                               visible_lens, out):
        """Bound FA planning lengths without synchronizing the device.

        CPU overlap lengths may over-estimate the committed prefix. The exact
        visible length is sawtooth-shaped at page boundaries, so align only on
        the GPU and use its monotonic envelope here. Use THIS pool's page size:
        the target CLI page (64) becomes a draft logical page (256) under DCP4.
        """
        source = prefix_lens_cpu if prefix_lens_cpu is not None else reserved_lens_cpu
        if source is None:
            out.copy_(visible_lens.to("cpu"))
        else:
            out.copy_(torch.clamp(source, max=self.window + self.page_size - 1))

    def physical_slots(self, request_ids, positions):
        return request_ids.to(torch.int64) * self.ring_stride + positions.to(torch.int64) % self.ring_stride

    def _put_gpu(self, physical, payload):
        for layer in range(self.layer_num):
            self.k_buffer[layer].index_copy_(0, physical, payload[:, layer, 0])
            self.v_buffer[layer].index_copy_(0, physical, payload[:, layer, 1])

    def commit_context(self, *, virtual, requests, positions, payload, is_decode=False):
        """payload includes only committed rows; scratch/bonus never enter backing.

        The caller preserves original context-projection ordering before filtering.
        Long prefill may wrap a ring repeatedly: retain only its last stride per
        request, avoiding duplicate-destination writes on the GPU.
        """
        if not is_decode:
            # Includes chunked prefill, mixed batches and retraction readmission.
            # The scheduler may subsequently deduplicate/repoint a radix prefix.
            self.invalidate_continuity()
        if virtual.numel() == 0:
            return
        virtual = virtual.to(dtype=torch.int64)
        # Blocking D2H completes producer work BEFORE acquiring the host lock.
        ids_cpu = virtual.cpu()
        data_cpu = payload.cpu()
        if int(ids_cpu.min()) < 0 or int(ids_cpu.max()) >= self.logical_size + self.page_size:
            raise ValueError("Bounded draft context is outside target virtual capacity")
        with self.backing_lock:
            self.backing.index_copy_(0, ids_cpu, data_cpu)
            self.backing_valid[ids_cpu] = True
        self.logical_versions[virtual] += 1
        ends = torch.full((self.max_requests + 1,), -1, dtype=torch.int64, device=self.device)
        ends.scatter_reduce_(0, requests.to(torch.int64), positions.to(torch.int64), reduce="amax", include_self=True)
        keep = positions > ends[requests.to(torch.int64)] - self.ring_stride
        physical = self.physical_slots(requests[keep], positions[keep])
        ids = virtual[keep]
        self._put_gpu(physical, payload[keep])
        self.slot_virtual[physical] = ids
        self.slot_version[physical] = self.logical_versions[ids]
        if is_decode:
            self._continuity_ready = True

    def prepare_window(self, *, target_table, draft_table, request_ids, prefix_lens, block_size,
                       request_owners=None):
        """Populate a page-aligned visible suffix and return draft-only scratch IDs.

        Runs before model_runner.forward, never inside a CUDA graph. Persistent
        ring rows survive ordinary decode; only misses/relocations load from CPU.
        """
        if block_size != 8:
            raise ValueError("Bounded GLM53 draft requires block_size=8")
        if (self.layer_transfer_counter is not None
                and self.layer_transfer_counter.consumer_index >= 0):
            self.layer_transfer_counter.wait_until(self.layer_num - 1)
            # L2 callbacks write CPU backing before recording this event. A CUDA
            # wait alone does not stop the CPU from reading partially restored KV.
            if str(self.device).startswith("cuda"):
                torch.cuda.current_stream(self.device).synchronize()
        if self.seen_restore_epoch != self.restore_epoch:
            self.slot_virtual.fill_(-1)
            self.seen_restore_epoch = self.restore_epoch
            self.invalidate_continuity()
        req = request_ids.to(torch.int64)
        lens = prefix_lens.to(torch.int64)
        start = torch.clamp(lens - self.window, min=0)
        start = start // self.page_size * self.page_size
        visible = lens - start
        offsets = torch.arange(self.window + self.page_size, device=self.device)
        positions = start[:, None] + offsets
        mask = offsets[None, :] < visible[:, None]
        physical = self.physical_slots(req[:, None], positions)
        reused = self._can_reuse_window(request_owners, len(request_ids))
        if reused:
            # A successful previous prepare established the entire visible ring.
            # Continuous decode only appends committed positions. Its scratch
            # lies beyond the visible prefix and stride > window+page+block, so
            # scratch cannot overwrite the retained history. Active target pages
            # remain owned by this request; first use/repoint/restore uses the
            # exact virtual-ID/version validation below.
            self.window_reuses += 1
            if self.window_reuses == 1 or self.window_reuses % 4096 == 0:
                logger.info(
                    "GLM53 bounded draft windows: reused=%d, validated=%d; "
                    "continuous decode skips the window version scan",
                    self.window_reuses, self.window_checks,
                )
        else:
            self.window_checks += 1
            safe_positions = torch.where(mask, positions, 0)
            virtual = target_table[req[:, None], safe_positions].to(torch.int64)
            live_virtual, live_physical = virtual[mask], physical[mask]
            versions = self.logical_versions[live_virtual]
            missing = ((self.slot_virtual[live_physical] != live_virtual)
                       | (self.slot_version[live_physical] != versions))
            if bool(missing.any()):
                ids = live_virtual[missing].cpu()
                with self.backing_lock:
                    if not bool(self.backing_valid[ids].all()):
                        raise RuntimeError("Bounded draft window has unmaterialized KV; refusing stale/uninitialized data")
                    values = self.backing.index_select(0, ids)
                slots = live_physical[missing]
                self._put_gpu(slots, values.to(self.device))
                self.slot_virtual[slots] = live_virtual[missing]
                self.slot_version[slots] = versions[missing]
        # Unused entries point to the sentinel row; attention lengths mask them.
        draft_table[req[:, None], offsets[None, :]] = torch.where(mask, physical, 0).to(draft_table.dtype)
        block = torch.arange(block_size, device=self.device)
        scratch = self.physical_slots(req[:, None], lens[:, None] + block)
        draft_table[req[:, None], visible[:, None] + block] = scratch.to(draft_table.dtype)
        self.slot_virtual[scratch] = -1
        if self.fastpath_enabled:
            if not reused:
                self._remember_window_owners(request_owners, len(request_ids))
            # Another prepare is not trusted until accepted context is committed.
            self._continuity_ready = False
        return visible.to(torch.int32), scratch.reshape(-1)

    def clear_bounded_cache(self):
        self.invalidate_continuity()
        with self.backing_lock:
            self.backing_valid.zero_()
            self.restore_epoch += 1
        self.slot_virtual.fill_(-1)


def bounded_draft_host_pool_class():
    # Keep host transfer dependencies out of the worker's module import path.
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    class BoundedDraftHostPool(MHATokenToKVPoolHost):
        def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
            if io_backend != "direct" or self.layout != "layer_first":
                raise ValueError("Bounded draft L2 requires layer_first/direct")
            host_ids, virtual = host_indices.cpu().long(), device_indices.cpu().long()
            with device_pool.backing_lock:
                if not bool(device_pool.backing_valid[virtual].all()):
                    raise RuntimeError("HiCache attempted to persist uncommitted bounded draft KV")
                values = device_pool.backing.index_select(0, virtual)
                for layer in range(device_pool.layer_num):
                    self.k_buffer[layer].index_copy_(0, host_ids, values[:, layer, 0])
                    self.v_buffer[layer].index_copy_(0, host_ids, values[:, layer, 1])

        def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend, *, is_draft=False):
            if io_backend != "direct" or self.layout != "layer_first":
                raise ValueError("Bounded draft L2 requires layer_first/direct")
            host_ids, virtual = host_indices.cpu().long(), device_indices.cpu().long()
            with device_pool.backing_lock:
                device_pool.backing[virtual, layer_id, 0] = self.k_buffer[layer_id].index_select(0, host_ids)
                device_pool.backing[virtual, layer_id, 1] = self.v_buffer[layer_id].index_select(0, host_ids)
                if layer_id == device_pool.layer_num - 1:
                    device_pool.backing_valid[virtual] = True
                    device_pool.restore_epoch += 1

    return BoundedDraftHostPool
