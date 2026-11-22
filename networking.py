"""Use the operating system trust store in this application, with TLS verification on."""
from __future__ import annotations
import os

def configure_network_tls():
    if os.name == 'nt':
        # Enterprise/Windows roots remain validated; no verify=False or unverified contexts.
        import truststore
        truststore.inject_into_ssl()
