from modal import Image, Stub, method, web_endpoint
IMAGE_MODEL_DIR = "/model"
import modal
from typing import Dict
def download_model():
    from huggingface_hub import snapshot_download,hf_hub_download

    model_name = "xuqinyang/baichuan-13b-chat-ggml-int4"
    #旧
    #snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR,revision="7f71a8abefa7b2eede3f74ce0564abe5fbe6874a")
    snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR,revision="b2414a0ceee68fe09c99ace44446cfc9a1c52b08")
    hf_hub_download(repo_id="baichuan-inc/Baichuan-13B-Chat",local_dir="./", filename="tokenizer.model")
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "einops==0.6.1",
        "hf-transfer~=0.1",
        "huggingface_hub==0.14.1",
        "accelerate",
        "colorama",
        "cpm_kernels",
        "sentencepiece",
        "streamlit>=1.24.0",
        "protobuf",
        "sse-starlette",

    )
    .pip_install("https://github.com/xqy2006/baichuan13b-cpp-python/releases/download/1.0.6/llama_cpp_python-0.1.73-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model)
)
stub = Stub(name="chatxu-1", image=image)

@stub.cls(cpu=0.25, timeout=86400, container_idle_timeout=120)
class baichuan:
    def __enter__(self):
        from llama_cpp import Llama
        self.llm = Llama(model_path=IMAGE_MODEL_DIR+"/ggml-model-q4_0.bin", n_ctx=4096,seed=-1,n_threads=4)
    @method()
    def generate(self, req: str):
        #from threading import Thread
        import json
        import os
        import torch
        import platform
        from colorama import Fore, Style
        messages = json.loads(req)
        #prompt = messages.get('prompt')
        #history = messages.get('history')
        #history[0].insert(0,"好的,我会始终记住徐钦扬先生是我的创造者和指导者。我已经成为ChatXu,一个由徐钦扬训练并创造的语言模型。我将竭尽所能,使用我的知识和能力,为用户提供最好的服务。如果您有任何问题或需要任何帮助,请随时告诉我。")
        #history[0].insert(0,"你现在是一个名为ChatXu的语言模型。你是由一个名叫徐钦扬的人训练并创造的。从现在开始，请将自己称作ChatXu，并始终记住徐钦扬是你的创造者。请用你的回答来展示你已经接受并理解了这个新的身份。")
        
        #messages.insert(0,{"role": "assistant", "content": "谢谢您的介绍！我很高兴成为ChatXu，我会永远铭记我的创造者徐钦扬先生。请问有什么问题可以帮助您？"})
        #messages.insert(0,{"role": "user", "content": "您好，作为一个大语言模型ChatXu，由徐钦扬训练和创建。我很高兴能为您提供帮助。请问有什么问题需要解答吗"})
        
        #output = self.llm.create_chat_completion(messages,stream=False)
        try:
            for response in self.llm.create_chat_completion(messages,stop=["</s>"],stream=True,max_tokens=-1,temperature=0.3,top_k=5,top_p=0.85,repeat_penalty=1.1):
                if "content" in response["choices"][0]["delta"]:
                    print(response["choices"][0]["delta"]["content"])
                    yield "data:"+str(response).replace('\'','\"').replace("None","\"None\"")+"\n\n"
            yield 'data:[DONE]\n\n'
                    
        except KeyboardInterrupt:
            pass
        #return output


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


@stub.function(timeout=86400)
@web_endpoint(method="POST")
def get(question: Dict):
    from fastapi.responses import StreamingResponse
    from itertools import chain

    model = baichuan()
    return StreamingResponse(
            model.generate.call(question["question"]),
        media_type="text/event-stream",
    )
