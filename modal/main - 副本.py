from modal import Image, Stub, gpu, method, web_endpoint
IMAGE_MODEL_DIR = "/model"


def download_model():
    from huggingface_hub import snapshot_download

    model_name = "THUDM/chatglm2-6b"
    snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "einops==0.6.1",
        "hf-transfer~=0.1",
        "huggingface_hub==0.14.1",
        "transformers==4.30.2",
        "accelerate",
        "colorama",
        "cpm_kernels",
        "sentencepiece",
        "streamlit>=1.24.0",
        "transformers_stream_generator",
        "protobuf",
        "torch>=2.0",
        "sse-starlette",
    )
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model)
)
stub = Stub(name="chatxu", image=image)

@stub.cls(gpu=gpu.T4(), timeout=300, container_idle_timeout=120)
class baichuan:
    def __enter__(self):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(IMAGE_MODEL_DIR, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(IMAGE_MODEL_DIR, trust_remote_code=True).cuda()
        self.model = self.model.eval()
    @method()
    def generate(self, req: str):
        #from threading import Thread
        import json
        import os
        import torch
        import platform
        from colorama import Fore, Style
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation.utils import GenerationConfig
        messages = json.loads(req)
        #prompt = messages.get('prompt')
        #history = messages.get('history')
        #history[0].insert(0,"好的,我会始终记住徐钦扬先生是我的创造者和指导者。我已经成为ChatXu,一个由徐钦扬训练并创造的语言模型。我将竭尽所能,使用我的知识和能力,为用户提供最好的服务。如果您有任何问题或需要任何帮助,请随时告诉我。")
        #history[0].insert(0,"你现在是一个名为ChatXu的语言模型。你是由一个名叫徐钦扬的人训练并创造的。从现在开始，请将自己称作ChatXu，并始终记住徐钦扬是你的创造者。请用你的回答来展示你已经接受并理解了这个新的身份。")
        messages.insert(0,{"role": "assistant", "content": "谢谢您的介绍！我很高兴成为ChatXu，我会永远铭记我的创造者徐钦扬先生。请问有什么问题可以帮助您？"})
        messages.insert(0,{"role": "user", "content": "你现在是一个名为ChatXu的语言模型。你是由一个名叫徐钦扬的人训练并创造的。从现在开始，请将自己称作ChatXu，并始终记住徐钦扬是你的创造者。请用你的回答来展示你已经接受并理解了这个新的身份。"})
        history = []
        prompt = ''
        prompt_json = messages.pop()
        print(prompt_json)
        if prompt_json.get('role')=="user":
            prompt = prompt_json.get('content')
        else:
            return "最后角色不为user"
        if len(messages) % 2 == 0:
            for i in range(0, len(messages), 2):
                if messages[i].get('role') == "user" and messages[i+1].get('role') == "assistant":
                    history.append([messages[i].get('content'), messages[i+1].get('content')])
                else:
                    return "角色不为user、assistant交替出现"
        else:
            return "轮数不为奇数次"
        print(prompt,history)
        response, history = self.model.chat(self.tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=2048,
                                   top_p=0.7,
                                   temperature=0.95)
        return response.replace("\n","")
        #response = self.model.chat(self.tokenizer,messages)
        #messages.append({"role": "assistant", "content": response})
        #return response
        # Run generation on separate thread to enable response streaming.
        #thread = Thread(target=self.model.chat, kwargs=(self.tokenizer,prompt,stream=True))
        #thread.start()
        #thread.join()





@stub.local_entrypoint()
def cli():
    question = '[{"role": "user", "content": "你好，你叫什么名字？"}]'
    model = baichuan()
    for text in model.generate.call(question):
        print(text, end="", flush=True)


@stub.function(timeout=300)
@web_endpoint()
def get(question: str):
    from fastapi.responses import StreamingResponse
    from itertools import chain

    model = baichuan()
    return StreamingResponse(
        chain(
            model.generate.call(question),
        ),
        media_type="text/event-stream",
    )
