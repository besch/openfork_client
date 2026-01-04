"""
Typed Configuration Classes for DGN Client

Provides type-safe configuration with dataclasses for better IDE support,
documentation, and validation.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import os


AcceptPolicy = Literal["all", "mine", "project", "users"]


@dataclass
class ProviderConfig:
    """Configuration for a DGN provider node."""
    
    # Job acceptance policy
    accept_policy: AcceptPolicy = "all"
    
    # Allowed targets for project/users policies
    allowed_targets: List[str] = field(default_factory=list)
    
    # Service filter (comma-separated or "auto")
    workflow_filter: str = "auto"
    
    # Specific service to run (for headless single-service mode)
    service: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_policies = ("all", "mine", "project", "users")
        if self.accept_policy not in valid_policies:
            raise ValueError(f"accept_policy must be one of {valid_policies}, got: {self.accept_policy}")
        
        # project/users policies require allowed_targets
        if self.accept_policy in ("project", "users") and not self.allowed_targets:
            raise ValueError(f"accept_policy '{self.accept_policy}' requires allowed_targets")


@dataclass
class AuthConfig:
    """Authentication configuration for DGN client."""
    
    # OAuth tokens (for Electron/web mode)
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    
    # API key (for headless mode)
    dgn_api_key: Optional[str] = None
    
    @property
    def is_headless_mode(self) -> bool:
        """Check if running in headless (API key) mode."""
        return self.dgn_api_key is not None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if any authentication method is configured."""
        return self.dgn_api_key is not None or self.access_token is not None


@dataclass
class DockerConfig:
    """Docker-related configuration."""
    
    # Skip Docker operations (for cloud/container environments)
    skip_docker: bool = False
    
    # Docker image pull timeout in seconds
    pull_timeout: int = 1800  # 30 minutes
    
    # Maximum concurrent Docker image downloads
    max_concurrent_downloads: int = 1
    
    # ComfyUI port for communication
    comfyui_port: int = 8188
    
    @classmethod
    def from_environment(cls) -> "DockerConfig":
        """Create DockerConfig from environment variables."""
        return cls(
            skip_docker=os.environ.get("SKIP_DOCKER", "").lower() in ("1", "true", "yes"),
            pull_timeout=int(os.environ.get("DOCKER_PULL_TIMEOUT", "1800")),
            max_concurrent_downloads=int(os.environ.get("DOCKER_MAX_DOWNLOADS", "1")),
            comfyui_port=int(os.environ.get("COMFYUI_PORT", "8188")),
        )


@dataclass 
class ClientConfig:
    """Complete DGN client configuration."""
    
    # Orchestrator URL
    orchestrator_url: str = "https://www.openfork.video"
    
    # Provider settings
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    
    # Authentication settings
    auth: AuthConfig = field(default_factory=AuthConfig)
    
    # Docker settings
    docker: DockerConfig = field(default_factory=DockerConfig)
    
    # Development mode
    dev_mode: bool = False
    
    @classmethod
    def from_environment(cls) -> "ClientConfig":
        """Create full ClientConfig from environment variables."""
        from config import DEV_MODE, ORCHESTRATOR_URL_PROD, ORCHESTRATOR_URL_DEV, HEADLESS_MODE
        
        dev_mode = DEV_MODE
        orchestrator_url = ORCHESTRATOR_URL_DEV if dev_mode else ORCHESTRATOR_URL_PROD
        
        # Parse provider config from environment
        accept_policy = os.environ.get("ACCEPT_POLICY", "all")
        allowed_targets_str = os.environ.get("ALLOWED_TARGETS", "")
        allowed_targets = [t.strip() for t in allowed_targets_str.split(",") if t.strip()]
        
        provider = ProviderConfig(
            accept_policy=accept_policy,  # type: ignore
            allowed_targets=allowed_targets,
            workflow_filter=os.environ.get("WORKFLOW_FILTER", "auto"),
            service=os.environ.get("SERVICE"),
        )
        
        auth = AuthConfig(
            access_token=os.environ.get("ACCESS_TOKEN"),
            refresh_token=os.environ.get("REFRESH_TOKEN"),
            dgn_api_key=os.environ.get("DGN_API_KEY"),
        )
        
        docker = DockerConfig.from_environment()
        # In headless mode, skip Docker operations (ComfyUI runs in same container)
        # and enforce 'all' policy (headless clients serve the public network)
        if HEADLESS_MODE:
            docker.skip_docker = True
            provider.accept_policy = "all"
            provider.allowed_targets = []
        
        return cls(
            orchestrator_url=orchestrator_url,
            provider=provider,
            auth=auth,
            docker=docker,
            dev_mode=dev_mode,
        )
