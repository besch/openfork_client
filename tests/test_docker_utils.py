import io
import tarfile
import threading

import pytest
import requests

from services import docker_utils


def _archive_with_file(name: str, content: bytes) -> bytes:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode='w') as archive:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return archive_bytes.getvalue()


class StreamResetOnceContainer:
    def __init__(self, archive_bytes: bytes):
        self.archive_bytes = archive_bytes
        self.calls = 0

    def get_archive(self, source_path):
        self.calls += 1
        if self.calls == 1:
            return self._resetting_stream(), {}
        return iter([self.archive_bytes[:11], self.archive_bytes[11:]]), {}

    def _resetting_stream(self):
        yield self.archive_bytes[:11]
        raise requests.exceptions.ConnectionError(
            ConnectionResetError(10054, "An existing connection was forcibly closed")
        )


class MissingFileContainer:
    def __init__(self):
        self.calls = 0

    def get_archive(self, source_path):
        self.calls += 1
        raise FileNotFoundError(source_path)


def test_copy_file_from_container_api_retries_transient_stream_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_utils, "_DOCKER_COPY_RETRY_BASE_SECONDS", 0)
    archive_bytes = _archive_with_file("result.png", b"generated-output")
    container = StreamResetOnceContainer(archive_bytes)
    dest_path = tmp_path / "result.png"

    docker_utils.copy_file_from_container_api(
        container,
        "/app/output/result.png",
        str(dest_path),
        threading.Event(),
    )

    assert dest_path.read_bytes() == b"generated-output"
    assert container.calls == 2
    assert not list(tmp_path.glob("*.part"))


def test_copy_file_from_container_api_does_not_retry_missing_source(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_utils, "_DOCKER_COPY_RETRY_BASE_SECONDS", 0)
    container = MissingFileContainer()
    dest_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        docker_utils.copy_file_from_container_api(
            container,
            "/app/output/missing.png",
            str(dest_path),
            threading.Event(),
        )

    assert container.calls == 1
    assert not dest_path.exists()
