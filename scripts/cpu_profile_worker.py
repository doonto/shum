#!/usr/bin/env python3
"""Profile the existing CPU ASR adapters without changing their decoding."""
from __future__ import annotations

import dataclasses
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters/asr"))


def event(phase, started, **values):
    from flix_fast_worker import normalize_diagnostics
    data, _ = normalize_diagnostics(
        {"event": "shum.profile", "version": 1, "phase": phase,
         "elapsed_seconds": time.monotonic() - started, **values})
    print(json.dumps(data,
                     allow_nan=False), file=sys.stderr, flush=True)


def configure_pools(configure):
    def bounded(threads):
        configure(threads)
        for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS"):
            os.environ[variable] = "1"
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    return bounded


class ObservedPrime:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def recognize(self, path):
        started = time.monotonic()
        result = self.model.recognize(path)
        event("inference", started, input=str(path),
              generation_status="not_exposed_by_engine")
        return result


class ObservedTurbo:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def transcribe(self, path, **parameters):
        started = time.monotonic()
        segments, info = self.model.transcribe(path, **parameters)

        def observed():
            raw = []
            complete = False
            try:
                for segment in segments:
                    raw.append({"start": segment.start, "end": segment.end,
                                "tokens": len(segment.tokens),
                                "temperature": segment.temperature})
                    yield segment
                complete = True
            finally:
                event("inference", started, input=str(path), segments=raw,
                      generator_consumed=complete,
                      decoder_token_limit=self.model.max_length,
                      generation_status="eos_not_exposed_by_engine",
                      transcription_options=dataclasses.asdict(info.transcription_options),
                      vad_options=dataclasses.asdict(info.vad_options),
                      duration_after_vad_seconds=info.duration_after_vad)

        return observed(), info


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("primeline", "turbo", "diarization"):
        raise ValueError("First argument must be primeline, turbo or diarization")
    component = sys.argv[1]
    arguments = sys.argv[2:]
    detailed = "--detail-profile" in arguments
    channels_last = "--embedding-channels-last" in arguments
    for flag in ("--detail-profile", "--embedding-channels-last"):
        if arguments.count(flag) > 1 or flag in arguments and component != "diarization":
            raise ValueError(f"{flag} is supported once, for diarization only")
    arguments = [argument for argument in arguments
                 if argument not in ("--detail-profile", "--embedding-channels-last")]
    if component == "diarization":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters/diarization"))
        import main as worker
        worker.configure_thread_environment = configure_pools(worker.configure_thread_environment)
        cleanup = ExitStack()

        def factory(directory, threads):
            import torch
            torch.set_num_interop_threads(1)
            pipeline = worker.default_pipeline_factory(directory, threads)
            if channels_last:
                from diarization_cpu_layout import install
                cleanup.callback(install(pipeline, event).close)
            if detailed:
                from diarization_cpu_observer import install
                cleanup.callback(install(pipeline, event).close)
            from threadpoolctl import threadpool_info
            event("observed_pools", STARTED, torch_intra_threads=torch.get_num_threads(),
                  torch_inter_threads=torch.get_num_interop_threads(),
                  native_pools=threadpool_info())
            return pipeline

        with cleanup:
            return worker.run(arguments, pipeline_factory=factory)
    if component == "primeline":
        import primeline_onnx_worker as worker
        original = worker.load_model
        wrapper = ObservedPrime
    else:
        import turbo_worker as worker
        original = worker.default_factory
        wrapper = ObservedTurbo
    worker.common.configure_thread_environment = configure_pools(
        worker.common.configure_thread_environment)

    def factory(directory, threads):
        event("startup_checks", STARTED, pid=os.getpid(), threads=threads,
              auxiliary_blas_threads=1, omp_max_active_levels=1)
        started = time.monotonic()
        model = original(directory, threads)
        event("model_load", started, pid=os.getpid(), threads=threads,
              compute_type="float32" if component == "primeline" else model.model.compute_type)
        return wrapper(model)

    return worker.run(arguments, factory=factory)


if __name__ == "__main__":
    sys.exit(main())
