import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM
import gradio as gr

load_dotenv()

model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

parameters = {
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.TEMPERATURE: 0.5,
}

watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
    apikey=os.getenv("WATSONX_APIKEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params=parameters,
)

def generate_response(prompt_txt):
    if not prompt_txt.strip():
        return "Please enter a question."
    return watsonx_llm.invoke(prompt_txt)

chat_application = gr.Interface(
    fn=generate_response,
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output", lines=6),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer.",
)

if __name__ == "__main__":
    chat_application.launch()