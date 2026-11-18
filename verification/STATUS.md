# Verification status

Full AI4Mars training and expert-set evaluation have not been run. No trained checkpoint or claimed real-data accuracy is bundled. The native model passed a CUDA forward/backward step; the default SMP backend still needs its dependencies and dataset downloads.

The independent repository test suite and synchronous Streamlit AppTest are executed before publishing. In this environment the UI test uses `--offline-test-loop` because local socket creation is blocked. That mocks only the unused asyncio event loop; it does not verify browser rendering.

The installed Torch/NumPy versions emit an ABI warning. Array conversion has a byte-buffer fallback; fresh installs use NumPy 1.26 as specified in requirements.txt.
