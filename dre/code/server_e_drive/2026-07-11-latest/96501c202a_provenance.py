"""Provenance constants for the Task 1 SEEGformer adaptation."""

UPSTREAM_REPOSITORY = "https://github.com/wcx7/SEEGformer"
UPSTREAM_COMMIT = "7a19723362d35adc0d463dfad8ad7a8b48ab6ee5"
UPSTREAM_LICENSE = "MIT"
UPSTREAM_COPYRIGHT = "Copyright (c) 2024 wcx7"
ADAPTATION_NOTE = (
    "Cross-patient Task 1 adaptation using three independent FFT branches "
    "(real, imaginary, amplitude), channel attention, and direct NEZ supervision."
)
