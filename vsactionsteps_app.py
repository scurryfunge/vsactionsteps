from fastapi import FastAPI
import torch
from transformers import pipeline, LlamaTokenizer, LlamaForCausalLM
from ray import serve
from ray.serve.handle import DeploymentHandle
from my_config import read_token
import os
from pathlib import Path
import ast

app = FastAPI()

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, vsactionstep_model_handle: DeploymentHandle) -> None:
        self.handle = vsactionstep_model_handle

    @app.get("/predict")
    async def predict(self, text: str):
        print("Making prediction ...")
        return await self.handle.predict.remote(text)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 8},
)
class VSActionStepModel:
    def __init__(self):
        home = Path.home()
        torchtrainer_path = os.path.join(str(home), "dev/testgpt/finetuning/llama2/TorchTrainer_9c9ed_00000_0_2024-05-20_01-47-38")
        checkpoint_path = os.path.join(torchtrainer_path, "checkpoint_000009/checkpoint")
        model = LlamaForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16, device_map="auto", token=read_token)
        tokenizer = LlamaTokenizer.from_pretrained(checkpoint_path, torch_dtype=torch.float16, device_map="auto", token=read_token)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Is CUDA available? {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        self.pipe = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
        )
    
    def generate_prompt(self, text):
        if text["test_requirement"]:
            return f"""Below is a description of a test that a verification engineer would like to carry out on a radio unit, along with what requirements should be covered by the test. Write a response that appropriately completes the request.

    ### VS title:
    {text["vs_title"]}

    ### Test description:
    {text["test_description"]}

    ### Requirements:
    {text["test_requirement"]}

    ### Verification specification action steps:"""
        else:
            return f"""Below is a description of a test that a verification engineer would like to carry out on a radio unit. Write a response that appropriately completes the request.

    ### VS title:
    {text["vs_title"]}
            
    ### Test description:
    {text["test_description"]}

    ### Verification specification action steps:"""
    
    def predict(self, text: str):
        text_dict = ast.literal_eval(text)
        prompt = self.generate_prompt(text_dict)
        print(f"prompt: {prompt}")
        return self.pipe(prompt, do_sample=True, return_full_text=False)[0]['generated_text']


entrypoint = APIIngress.bind(VSActionStepModel.bind())