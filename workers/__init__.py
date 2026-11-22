
def runtime_probe(events):
    """Lightweight packaging probe; does not open the microphone."""
    events.put('worker-ready')
