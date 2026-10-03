#!/usr/bin/env python3
"""Shared GGUF `general.*` metadata for the stems.cpp converters.

Implements the naming convention in docs/DISTRIBUTION.md, the same one sa3.cpp uses, so every
GGUF carries the basename, size label, version and license the Hugging Face cards and the
download scripts rely on. `general.architecture` is still set by each converter's
GGUFWriter(arch=...) call, and the runtime keys off it; this only adds catalog metadata.
"""

VERSION = "v1.0"


def size_label(n_params, n_models=1):
    """Parameter-count class per the GGUF convention, as in sa3.cpp: 100M and up render as
    '0.xB', smaller models as 'NM'. A bag of n models is 'nx<per-model>', the convention's
    form for groups of identical experts, e.g. htdemucs_ft -> '4x42M'."""
    def one(n):
        if n >= 1e8:
            return f"{n / 1e9:.1f}B".replace(".0B", "B")
        if n >= 1e5:
            return f"{n / 1e6:.0f}M"
        return f"{n / 1e3:.0f}K"
    return one(n_params) if n_models == 1 else f"{n_models}x{one(n_params / n_models)}"


def filename(basename, label, encoding):
    """<BaseName>-<SizeLabel>-<Version>-<Encoding>.gguf, e.g. htdemucs-42M-v1.0-F32.gguf."""
    return f"{basename}-{label}-{VERSION}-{encoding}.gguf"


def add_general(w, basename, n_params, license_id, n_models=1):
    """Stamp the catalog metadata. basename is the runtime's model id (htdemucs,
    mel_band_roformer_kim, ...), which is also general.name and stems.model."""
    w.add_name(basename)
    w.add_string("general.basename", basename)
    w.add_string("general.size_label", size_label(n_params, n_models))
    w.add_string("general.version", VERSION)
    w.add_string("general.license", license_id)


def add_source(w, name, organization, repo_url, version, source_files=None):
    """Record the exact upstream checkpoint with the standard base-model fields, so a
    standalone GGUF still says where its weights came from."""
    w.add_base_model_count(1)
    w.add_base_model_name(0, name)
    w.add_base_model_organization(0, organization)
    w.add_base_model_repo_url(0, repo_url)
    w.add_base_model_version(0, version)
    if source_files:
        w.add_string("general.source.file", ", ".join(source_files))
