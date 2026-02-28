import os
import re
from openai import OpenAI
from utils.constants import (
    EMBEDDING_MODEL, 
    LLM_MODEL, 
    RAG,
    SILICONFLOW_API_KEY,
    DASHSCOPE_API_KEY,
    OPENAI_API_KEY,
    OLLAMA_API_KEY,
    SILICONFLOW_BASE_URL,
    DASHSCOPE_BASE_URL,
    OLLAMA_BASE_URL
)
from utils.utility_functions import log_llm_response, log_update
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

if LLM_MODEL == "deepseek-ai/DeepSeek-V3":
# Environment variables
    spec_client = OpenAI(
        api_key=SILICONFLOW_API_KEY, 
        base_url=SILICONFLOW_BASE_URL
    )

    client = OpenAI(
        api_key=SILICONFLOW_API_KEY, 
        base_url=SILICONFLOW_BASE_URL
    )
elif LLM_MODEL == "qwen-plus-0428":
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY, 
        base_url=DASHSCOPE_BASE_URL
    )

    spec_client = OpenAI(
        api_key=DASHSCOPE_API_KEY, 
        base_url=DASHSCOPE_BASE_URL
    )
elif LLM_MODEL == "o1-preview":
    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    spec_client = OpenAI(
        api_key=OPENAI_API_KEY
    )
elif LLM_MODEL == "qwen3-8b-ft-0826":
    client = OpenAI(
        api_key=OLLAMA_API_KEY,
        base_url=OLLAMA_BASE_URL
    )

    spec_client = OpenAI(
        api_key=OLLAMA_API_KEY,
        base_url=OLLAMA_BASE_URL
    )
elif LLM_MODEL == "llm-4o":
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )


extra_body = {
    "enable_thinking": False,
}


embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def request_llm_rag(system_content, user_contents, assistant_content, temperature):
    log_update("[GPTR] Using RAG")
    print("[GPTR] Using RAG")
    vectorstore = FAISS.load_local("vectorstore_2", embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        model=LLM_MODEL, 
        temperature=temperature
    )

    last_user_content = user_contents[-1]
    user_contents = user_contents[:-1]

    prompt = [
        ("system", system_content + "\n\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
    
    history = []
    if assistant_content:
        for i in range(max(len(user_contents), len(assistant_content))):
            if i < len(user_contents):
                history.append(HumanMessage(content=user_contents[i]))
            if i < len(assistant_content):
                history.append(AIMessage(content=assistant_content[i]))
    else:
        for content in user_contents:
            history.append(HumanMessage(content=content))

    prompt_template = ChatPromptTemplate.from_messages(prompt)
    qna_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, qna_chain)
    
    inputs = {
        "history": history,
        "input": last_user_content
    }
    response = rag_chain.invoke(inputs)
    answer = response.get("answer", "")
    
    messages = [("system", system_content)]
    if assistant_content:
        for i in range(max(len(user_contents), len(assistant_content))):
            if i < len(user_contents):
                messages.append(("user", user_contents[i]))
            if i < len(assistant_content):
                messages.append(("assistant", assistant_content[i]))
    else:
        for content in user_contents:
            messages.append(("user", content))
    messages.append(("user", last_user_content))
    log_llm_response(messages, answer)
    
    matches = re.match(r"(.*?)```(.*?)```(.*)", answer, re.DOTALL)
    
    if matches:
        return matches
    else:
        with open("invalid_assistant_reply.txt", "a") as file:
            file.write(answer + "\n\n" + "-" * 150 + "\n\n")
        return None


def request_llm(system_content, user_contents, assistant_content, temperature):
    if RAG:
        return request_llm_rag(system_content, user_contents, assistant_content, temperature)

    messages = [{"role": "user", "content": system_content}]

    if assistant_content:
        for i in range(max(len(user_contents), len(assistant_content))):
            if i < len(user_contents):
                messages.append({"role": "user", "content": user_contents[i]})
            if i < len(assistant_content):
                messages.append({"role": "assistant", "content": assistant_content[i]})
    else:
        for content in user_contents:
            messages.append({"role": "user", "content": content})


    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        extra_body=extra_body if "qwen" in LLM_MODEL else None
    )
    assistant_reply = completion.choices[0].message.content
    matches = []
    reasoning = re.search(r"<reasoning>(.*?)</reasoning>", assistant_reply, re.DOTALL)
    config = re.search(r"<config>(.*?)</config>", assistant_reply, re.DOTALL)

    if config != None:
        if reasoning != None:
            matches.append(reasoning.group(1).strip())
        else:
            matches.append(" ")
        matches.append(config.group(1).strip())
        matches.append(' ')
    else:
        matches = None

    log_llm_response(messages, assistant_reply)

    if matches is not None:
        return matches 

    with open("invalid_assistant_reply.txt", "a") as file:
        file.write(assistant_reply + "\n\n" + "-" * 150 + "\n\n")
    return None

def send_llm_request(system_contents, user_contents, temperature):
    messages = [{"role": "user", "content": system_contents}, 
                {"role": "user", "content": user_contents}]

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        extra_body=extra_body if "qwen" in LLM_MODEL else None
    )

    assistant_reply = completion.choices[0].message.content
    log_llm_response(messages, assistant_reply)

    return assistant_reply