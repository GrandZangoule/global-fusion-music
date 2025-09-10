import gradio as gr

def add(x, y): 
    return x + y

gr.Interface(fn=add, inputs=["number", "number"], outputs="number").launch(share=True)