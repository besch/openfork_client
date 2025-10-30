'''
Configuration for the DGN Client
'''
import os
import sys
from dotenv import load_dotenv


load_dotenv()

class Config:
    # --- General Configuration ---
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # os.chdir(ROOT_DIR) # This should not be here, it changes the working directory for the entire process

    CACHE_DIR = os.path.join(ROOT_DIR, '.cache')
    DEV_MODE = False

    # --- Supabase Configuration ---
    SUPABASE_URL = "https://vmuylzvwqravkmdmcpgv.supabase.co"
    SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtdXlsenZ3cXJhdmttZG1jcGd2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIxNDM3MjAsImV4cCI6MjA2NzcxOTcyMH0.f2USQOkuKhPksSLSXhTlyl5zTstyCyYvzdiHV9HQUKw"

    # --- Orchestrator Configuration ---
    ORCHESTRATOR_URL_PROD = os.getenv("ORCHESTRATOR_URL_PROD", "https://www.openfork.video")
    ORCHESTRATOR_URL_DEV = os.getenv("ORCHESTRATOR_URL_DEV", "http://localhost:3000")

    # --- Docker Image Configuration ---
    # Maps a service type to a full Docker Hub image name.
    DOCKER_HUB_USERNAME = "beschiak"