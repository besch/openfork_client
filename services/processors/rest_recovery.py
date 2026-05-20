"""
Helpers for REST processors whose API container can exit after writing output.
"""

import logging
import os
import posixpath
import shutil
import tarfile
import tempfile
import time
from typing import Dict, Iterable, Optional

import requests

from services.docker_manager import docker_manager
from services.processors.base import BaseJobProcessor


def get_processor_service_type(processor: BaseJobProcessor) -> Optional[str]:
    service_type = processor.job.get("service_type")
    if service_type:
        return service_type
    try:
        return processor.client.get_service_type_for_workflow(processor.workflow_type)
    except Exception:
        return None


def clean_container_exit_detected(processor: BaseJobProcessor) -> bool:
    service_type = get_processor_service_type(processor)
    if not service_type:
        return False

    try:
        container_name = docker_manager.get_container_name(service_type)
        container = docker_manager.client.containers.get(container_name)
        container.reload()
        state = container.attrs.get("State", {})
        return (
            state.get("Status") in {"exited", "dead"}
            and state.get("ExitCode") == 0
            and not state.get("OOMKilled", False)
        )
    except Exception as exc:
        logging.debug("Could not inspect REST container state: %s", exc)
        return False


def poll_rest_job_with_clean_exit(
    processor: BaseJobProcessor,
    api_base_url: str,
    remote_job_id: str,
    *,
    poll_interval: int,
    max_wait_time: int,
    service_label: str,
    session: Optional[requests.Session] = None,
    completed_statuses: Iterable[str] = ("completed",),
    failed_statuses: Iterable[str] = ("failed",),
    status_path_template: str = "/status/{job_id}",
    require_seen_job: bool = True,
) -> Dict:
    start_time = time.time()
    saw_remote_job = False
    consecutive_connection_errors = 0
    http = session or requests
    completed = set(completed_statuses)
    failed = set(failed_statuses)

    while time.time() - start_time < max_wait_time:
        if processor.is_cancelled():
            return {"status": "cancelled", "error": "Shutdown requested"}

        try:
            response = http.get(
                f"{api_base_url}{status_path_template.format(job_id=remote_job_id)}",
                timeout=10,
            )
            consecutive_connection_errors = 0

            if response.status_code == 200:
                saw_remote_job = True
                data = response.json()
                status = data.get("status")

                if status in completed:
                    logging.info("%s remote job %s completed", service_label, remote_job_id)
                    return data
                if status in failed:
                    logging.error(
                        "%s remote job %s failed: %s",
                        service_label,
                        remote_job_id,
                        data.get("error"),
                    )
                    return data
                logging.debug("%s remote job %s status: %s", service_label, remote_job_id, status)
            else:
                logging.warning("%s status check returned %s", service_label, response.status_code)
        except requests.exceptions.RequestException as exc:
            consecutive_connection_errors += 1
            logging.warning("%s status check failed: %s", service_label, exc)
            if (
                (saw_remote_job or not require_seen_job)
                and consecutive_connection_errors >= 3
                and clean_container_exit_detected(processor)
            ):
                logging.warning(
                    "%s API exited cleanly before final status for remote job %s; "
                    "attempting output recovery from container.",
                    service_label,
                    remote_job_id,
                )
                return {
                    "status": "completed",
                    "recovered_from_clean_container_exit": True,
                }

        processor.shutdown_event.wait(poll_interval)

    return {
        "status": "failed",
        "error": f"Timeout waiting for {service_label} generation",
    }


def recover_output_from_clean_container_exit(
    processor: BaseJobProcessor,
    local_path: str,
    *,
    container_output_path: str,
    extensions: Optional[Iterable[str]] = None,
    prefer_name: Optional[str] = None,
    require_clean_exit: bool = True,
) -> Optional[str]:
    if require_clean_exit and not clean_container_exit_detected(processor):
        return None

    service_type = get_processor_service_type(processor)
    if not service_type:
        return None

    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    normalized_extensions = tuple(ext.lower() for ext in (extensions or ()))

    try:
        if _copy_from_container_archive(
            service_type,
            container_output_path,
            local_path,
            normalized_extensions,
            prefer_name,
        ):
            logging.info(
                "Recovered REST output for job %s from %s",
                processor.job_id,
                container_output_path,
            )
            return local_path
    except Exception as exc:
        logging.debug("Docker API output recovery failed: %s", exc)

    try:
        docker_manager.copy_file_from_container(
            service_type,
            container_output_path,
            local_path,
            processor.shutdown_event,
        )
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            logging.info(
                "Recovered REST output for job %s from %s",
                processor.job_id,
                container_output_path,
            )
            return local_path
    except Exception as exc:
        logging.warning("Could not recover REST output from container: %s", exc)

    return None


def _copy_from_container_archive(
    service_type: str,
    source_path: str,
    dest_path: str,
    extensions: tuple[str, ...],
    prefer_name: Optional[str],
) -> bool:
    client = getattr(docker_manager, "client", None)
    if client is None:
        return False

    container_name = docker_manager.get_container_name(service_type)
    container = client.containers.get(container_name)
    stream, _ = container.get_archive(source_path)

    temp_tar_path = None
    temp_dest_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as temp_tar:
            temp_tar_path = temp_tar.name
            for chunk in stream:
                if chunk:
                    temp_tar.write(chunk)

        with tarfile.open(temp_tar_path, mode="r:*") as archive:
            member = _select_output_member(
                archive,
                source_path,
                extensions,
                prefer_name,
            )
            extracted = archive.extractfile(member)
            if extracted is None:
                return False

            dest_dir = os.path.dirname(os.path.abspath(dest_path))
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=dest_dir,
                prefix=f".{os.path.basename(dest_path)}.",
                suffix=".part",
            ) as temp_dest:
                temp_dest_path = temp_dest.name
                with extracted:
                    shutil.copyfileobj(extracted, temp_dest)

        os.replace(temp_dest_path, dest_path)
        temp_dest_path = None
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    finally:
        for path in (temp_tar_path, temp_dest_path):
            if path and os.path.exists(path):
                os.remove(path)


def _select_output_member(
    archive: tarfile.TarFile,
    source_path: str,
    extensions: tuple[str, ...],
    prefer_name: Optional[str],
) -> tarfile.TarInfo:
    safe_name = posixpath.basename(source_path.rstrip("/"))

    members = [member for member in archive.getmembers() if member.isfile()]
    if extensions:
        members = [
            member
            for member in members
            if posixpath.basename(member.name).lower().endswith(extensions)
        ]
    if not members:
        raise FileNotFoundError(f"No matching output file found in '{source_path}'")

    exact_matches = [
        member
        for member in members
        if posixpath.basename(member.name.rstrip("/")) == safe_name
    ]
    if exact_matches:
        return max(exact_matches, key=lambda member: member.mtime)

    if prefer_name:
        preferred = [
            member
            for member in members
            if prefer_name in posixpath.basename(member.name)
        ]
        if preferred:
            return max(preferred, key=lambda member: member.mtime)

    return max(members, key=lambda member: member.mtime)
