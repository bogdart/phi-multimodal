https://huggingface.co/microsoft/Phi-4-multimodal-instruct-onnx

uv init
uv pip install -U "huggingface_hub[cli]"
uv pip install --pre onnxruntime-genai-cuda
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi4-mm.py -o phi4-mm.py
bash
huggingface-cli download microsoft/Phi-4-multimodal-instruct-onnx --include gpu/* --local-dir .

python phi4-mm.py -m gpu/gpu-int4-rtn-block-32 -e cuda