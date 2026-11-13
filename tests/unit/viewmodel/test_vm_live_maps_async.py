"""Unit tests for Phase 5 async QThreadPool offload in VMLiveMaps."""

import time
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from spectroview.model.spectra_store import MapData
from spectroview.model.spectral_parameters import (
    PARAM_RB_RATIO,
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
)
from spectroview.viewmodel.vm_live_maps import (
    ASYNC_SPECTRUM_THRESHOLD,
    VMLiveMaps,
)


def _make_2d_map(name: str = "big_2d", nx: int = 5, ny: int = 5) -> MapData:
    x = np.linspace(400.0, 1600.0, 61)
    n = nx * ny
    coords = np.array([[float(i % nx), float(i // nx)] for i in range(n)], dtype=np.float64)
    Y = np.zeros((n, len(x)), dtype=np.float32)
    for i in range(n):
        gx, gy = coords[i]
        Y[i] = (5.0 + gx) * np.exp(-((x - 700.0) ** 2) / (2 * 25.0 ** 2)) + (
            4.0 + gy
        ) * np.exp(-((x - 1300.0) ** 2) / (2 * 30.0 ** 2))
    return MapData(
        name=name,
        x0=x,
        Y0=Y,
        coords=coords,
        is_active=np.ones(n, dtype=bool),
        fnames=[f"s_{i}" for i in range(n)],
        colors=[None] * n,
        labels=[None] * n,
    )


def _make_1d_profile(name: str = "prof", n: int = 16, offset: float = 0.0) -> MapData:
    x = np.linspace(400.0, 1600.0, 61)
    coords = np.column_stack([np.linspace(-5.0, 5.0, n), np.zeros(n)])
    Y = np.zeros((n, len(x)), dtype=np.float32)
    for i in range(n):
        Y[i] = (3.0 + offset + 0.5 * i) * np.exp(-((x - 700.0) ** 2) / (2 * 25.0 ** 2))
    return MapData(
        name=name,
        x0=x,
        Y0=Y,
        coords=coords,
        is_active=np.ones(n, dtype=bool),
        fnames=[f"p_{i}" for i in range(n)],
        colors=[None] * n,
        labels=[None] * n,
    )


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    app = QApplication.instance()
    while time.perf_counter() < deadline:
        if predicate():
            return True
        if app is not None:
            app.processEvents()
        time.sleep(0.01)
    if app is not None:
        app.processEvents()
    return bool(predicate())


class TestVMLiveMapsAsyncOffload:
    """Verify threshold gating, parity with sync results, generation cancellation, and shutdown."""

    def test_default_threshold_is_25000(self, qapp):
        assert ASYNC_SPECTRUM_THRESHOLD == 25_000
        vm = VMLiveMaps(mode="map")
        try:
            assert vm._async_threshold == 25_000
        finally:
            vm.shutdown()

    def test_below_threshold_runs_synchronously(self, qapp):
        md = _make_2d_map(nx=4, ny=4)  # 16 spectra < 25,000
        vm = VMLiveMaps(mode="map", async_threshold=100)
        received = []
        vm.map_data_ready.connect(lambda *args: received.append(args))
        try:
            vm.set_selector_range(SELECTOR_RED, 600.0, 800.0, immediate=True, calculate=False)
            vm.set_selector_active(SELECTOR_RED, True, calculate=False)
            vm.set_map_data(md)
            assert not vm.is_async_busy()
            assert len(received) == 1
            assert received[0][0] == SELECTOR_RED
        finally:
            vm.shutdown()

    def test_at_or_above_threshold_offloads_and_matches_sync(self, qapp):
        md = _make_2d_map(nx=5, ny=5)  # 25 spectra

        vm_sync = VMLiveMaps(mode="map", async_threshold=10_000)
        vm_async = VMLiveMaps(mode="map", async_threshold=20)
        sync_out = []
        async_out = []
        vm_sync.map_data_ready.connect(lambda *args: sync_out.append(args))
        vm_async.map_data_ready.connect(lambda *args: async_out.append(args))

        try:
            for vm in (vm_sync, vm_async):
                vm.set_selector_range(SELECTOR_RED, 600.0, 800.0, immediate=True, calculate=False)
                vm.set_selector_range(SELECTOR_BLUE, 1200.0, 1400.0, immediate=True, calculate=False)
                vm.set_selector_parameter(SELECTOR_GREEN, PARAM_RB_RATIO)
                vm.set_selector_active(SELECTOR_RED, True, calculate=False)
                vm.set_selector_active(SELECTOR_BLUE, True, calculate=False)
                vm.set_selector_active(SELECTOR_GREEN, True, calculate=False)
                vm.set_map_data(md)

            assert len(sync_out) == 3
            assert _wait_until(lambda: len(async_out) == 3 and not vm_async.is_async_busy())

            assert [x[0] for x in async_out] == [x[0] for x in sync_out]
            for s, a in zip(sync_out, async_out):
                assert a[0] == s[0]
                np.testing.assert_allclose(a[1], s[1], equal_nan=True)
                assert a[2] == pytest.approx(s[2])
                assert a[3] == pytest.approx(s[3])
                assert a[4] == pytest.approx(s[4])
                assert a[5] == s[5]
        finally:
            vm_sync.shutdown()
            vm_async.shutdown()

    def test_multi_profile_async_parity(self, qapp):
        p1 = _make_1d_profile("p1", n=12, offset=0.0)
        p2 = _make_1d_profile("p2", n=12, offset=2.0)

        vm_sync = VMLiveMaps(mode="map", async_threshold=1000)
        vm_async = VMLiveMaps(mode="map", async_threshold=20)  # 24 >= 20
        sync_out = []
        async_out = []
        vm_sync.multi_profile_data_ready.connect(lambda *args: sync_out.append(args))
        vm_async.multi_profile_data_ready.connect(lambda *args: async_out.append(args))

        try:
            for vm in (vm_sync, vm_async):
                vm.set_selector_range(SELECTOR_RED, 600.0, 800.0, immediate=True, calculate=False)
                vm.set_selector_active(SELECTOR_RED, True, calculate=False)
                vm.set_map_datas([p1, p2], primary_name="p1")

            assert len(sync_out) == 1
            assert _wait_until(lambda: len(async_out) == 1 and not vm_async.is_async_busy())

            assert async_out[0][0] == sync_out[0][0]
            assert async_out[0][2] == sync_out[0][2]
            for se, ae in zip(sync_out[0][1], async_out[0][1]):
                assert ae["name"] == se["name"]
                np.testing.assert_allclose(ae["positions"], se["positions"])
                np.testing.assert_allclose(ae["values"], se["values"], equal_nan=True)
        finally:
            vm_sync.shutdown()
            vm_async.shutdown()

    def test_stale_generation_drops_superseded_payload(self, qapp):
        md = _make_2d_map(nx=5, ny=5)
        vm = VMLiveMaps(mode="map", async_threshold=20)
        received = []
        vm.map_data_ready.connect(lambda *args: received.append(args))

        try:
            vm.set_selector_range(SELECTOR_RED, 600.0, 800.0, immediate=True, calculate=False)
            vm.set_selector_active(SELECTOR_RED, True, calculate=False)
            vm.set_map_data(md)
            assert _wait_until(lambda: len(received) == 1)

            # Simulate an old-generation payload arriving after a newer range/clear bump
            received.clear()
            stale_gen = vm._generation_id - 1
            vm._on_async_batch_finished(
                stale_gen,
                {
                    "elapsed_ms": 1.0,
                    "outputs": [("map_2d", SELECTOR_RED, np.ones((2, 2)), [0, 1, 0, 1], 1.0, 1.0, "STALE")],
                    "cache_entries": {},
                    "md_versions": {id(md): (md.name, md.data_version)},
                },
            )
            assert received == []

            # Clear bumps generation so a late payload for cleared map is discarded too
            current_gen = vm._generation_id
            vm.set_map_datas([])
            assert vm._generation_id > current_gen
            vm._on_async_batch_finished(
                current_gen,
                {
                    "elapsed_ms": 1.0,
                    "outputs": [("map_2d", SELECTOR_RED, np.ones((2, 2)), [0, 1, 0, 1], 1.0, 1.0, "STALE")],
                    "cache_entries": {},
                    "md_versions": {id(md): (md.name, md.data_version)},
                },
            )
            assert received == []
        finally:
            vm.shutdown()

    def test_shutdown_prevents_subsequent_work_and_drains_cleanly(self, qapp):
        md = _make_2d_map(nx=5, ny=5)
        vm = VMLiveMaps(mode="map", async_threshold=10)
        received = []
        vm.map_data_ready.connect(lambda *args: received.append(args))

        vm.set_selector_range(SELECTOR_RED, 600.0, 800.0, immediate=True, calculate=False)
        vm.set_selector_active(SELECTOR_RED, True, calculate=False)
        vm.set_map_data(md)
        vm.shutdown(wait_ms=2000)

        assert not vm.is_async_busy()
        count_at_shutdown = len(received)
        vm.set_selector_range(SELECTOR_RED, 650.0, 750.0, immediate=True, calculate=True)
        qapp.processEvents()
        assert len(received) == count_at_shutdown
