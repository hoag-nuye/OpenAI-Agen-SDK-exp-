import litellm
model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"
if not litellm.supports_vision(model=model_name):
    print("Model không hỗ trợ vision.")
else:
    print("Model hỗ trợ vision.")
