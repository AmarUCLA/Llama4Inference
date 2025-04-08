import streamlit as st
import openai
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(
    page_title="Maverick Chat App",
    page_icon="💬",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "previous_upload_state" not in st.session_state:
    st.session_state.previous_upload_state = None

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

st.title("Maverick Chat App")
st.markdown("Chat with an LLM that can see and understand images")

with st.sidebar:
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploaded_image = None
        st.session_state.previous_upload_state = None
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user" and isinstance(message["content"], dict):
            st.write(message["content"]["text"])
            if "image_data" in message["content"]:
                st.image(message["content"]["image_data"], caption="Uploaded Image")
        else:
            st.write(message["content"])

uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.uploaded_image = {
        "data": image,
        "base64": encode_image_to_base64(image)
    }
    
    st.image(image, caption="Uploaded Image", use_container_width=False, width=300)
    
    st.session_state.previous_upload_state = True
elif st.session_state.previous_upload_state is True:
    st.session_state.uploaded_image = None
    st.session_state.uploaded_image = None
    st.session_state.previous_upload_state = False

if prompt := st.chat_input("Type your message here..."):
    user_message = {"role": "user"}
    user_message = {"role": "user"}
    
    if st.session_state.uploaded_image is not None:
        user_message["content"] = {
            "text": prompt,
            "image_data": st.session_state.uploaded_image["data"]
        }
        
        api_content = [
            {"type": "text", "text": prompt}
        ]
        
        api_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{st.session_state.uploaded_image['base64']}"
            }
        })
    else:
        user_message["content"] = prompt
        api_content = [{"type": "text", "text": prompt}]
    
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user"):
        if isinstance(user_message["content"], dict):
            st.write(user_message["content"]["text"])
            st.image(user_message["content"]["image_data"], caption="Uploaded Image")
        else:
            st.write(user_message["content"])
    
    st.session_state.uploaded_image = None
    st.session_state.previous_upload_state = False
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # CHANGE THE BASE_URL to the IP address of the server you are running the vllm server on 
            # Usually it is 127.0.0.1 or localhost
            client = openai.OpenAI(
                base_url="http://127.0.0.1:8080/v1",
                api_key="EMPTY",
            )
            
            message_history = []
            
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    if isinstance(msg["content"], dict):
                        message_history.append({"role": "user", "content": msg["content"].get("text", "")})
                    else:
                        message_history.append({"role": "user", "content": msg["content"]})
                else:
                    message_history.append({"role": "assistant", "content": msg["content"]})
            
            stream = client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    *message_history,
                    {"role": "user", "content": api_content}
                ],
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    st.rerun()
