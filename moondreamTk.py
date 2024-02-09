#from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer,  TextIteratorStreamer
import tkinter as tk; from tkinter import messagebox
from PIL import Image, ImageTk
from threading import Thread
import cv2, time
import numpy as np
from mss import mss

sct = mss()
screen_width, screen_height = sct.monitors[1]['width'], sct.monitors[1]['height']
model_running = change_detected = False
prev_frame = None
stabilization_time = 1200  # Time to wait for stabilization in milliseconds
last_change_time = 0
change_threshold = 50000
change_sensitivity = 10
auto_submit = False
image1_label = image2_label = stitched_image_label = None
import torch

capture_area = {'top': 174, 'left': 126, 'width': 610, 'height': 330}
last_pos = None  # To track the last position of the mouse
zoom_factor = 1.1  # Adjust the zoom factor as needed

# Download and load the model outside of the function
model_id = "vikhyatk/moondream1"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(0)
tokenizer = Tokenizer.from_pretrained(model_id)

cap = cv2.VideoCapture(0) # Initialize webcam capture
captured_image1 = captured_image2 = None # Global variables to store captured images

# Function to capture and store image
def capture_image(image_no):
    global captured_image1, captured_image2
    frame = prev_frame
    if image_no == 1:
        captured_image1 = frame
        update_label_image(image1_label, frame)
    elif image_no == 2:
        captured_image2 = frame
        update_label_image(image2_label, frame)

def update_label_image(label, image, size=(240, 180)):
    image = cv2.resize(image, size)
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk  # Keep a reference, prevent garbage-collection
    label.config(image=imgtk)

def on_compare():
    if captured_image1 is not None and captured_image2 is not None:
        box_width = int(0.02 * captured_image1.shape[1])
        cv2.rectangle(captured_image1, (0, 0), (captured_image1.shape[1], captured_image1.shape[0]), (0, 0, 0), box_width)
        cv2.rectangle(captured_image2, (0, 0), (captured_image2.shape[1], captured_image2.shape[0]), (0, 0, 0), box_width)
        blank_image = np.full((100, captured_image1.shape[1], 3), 255, dtype=np.uint8)
        stitched_image = np.vstack((captured_image1, blank_image, captured_image2))
        stitched_image  = cv2.resize(stitched_image , (480, 780))
        update_label_image(stitched_image_label, stitched_image, (120,195))
        compare_prompt_text = compare_prompt_entry.get("1.0", "end-1c")
        Thread(target=moondream, args=(compare_prompt_text, stitched_image)).start()
        
def moondream(prompt, frame, model=model, tokenizer=tokenizer):
    global model_running; model_running = True
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    enc_image = model.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    thread_args = (enc_image, prompt, tokenizer, "")
    thread_kwargs = {"streamer": streamer}
    thread = Thread(
                target=model.answer_question,
                args=thread_args,
                kwargs=thread_kwargs,
            )
    thread.start()
    
    buffer = ""
    for new_text in streamer:
        if not new_text.endswith("<") and not new_text.endswith("END"):
            buffer += new_text
            update_output(buffer)
        else:
            if new_text.endswith("<"):
                new_text = new_text[:-1]  # Remove the last character if it is "<"
            if new_text.endswith("END"):
                new_text = new_text[:-3]  # Remove the last 3 characters if they are "END"
            if new_text.endswith("<"):
                new_text = new_text[:-1]  # Remove the last character if it is "<"
            buffer += new_text
            update_output(buffer)
            break
    model_running = False

def update_output(text):
    output_text.delete('1.0', tk.END)
    output_text.insert(tk.END, text)

def on_submit():
    prompt_text = prompt_entry.get("1.0", "end-1c")
    if not model_running: Thread(target=moondream, args=(prompt_text,prev_frame)).start()

def update_image():
    global prev_frame, auto_submit, change_detected, last_change_time
    selected = selected_input.get()  # Get the current selection from the dropdown
    #global capture_area
    
    
    if selected == "Webcam":
        # Capture from the webcam
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (480, 340))
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif selected == "Screen Capture":
        # Capture from the screen
        sct_img = sct.grab(capture_area)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame = cv2.resize(frame, (480, 340))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret = True
    else:
        ret = False
    
    if ret:
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        webcam_label.imgtk = imgtk  # Keep a reference, prevent garbage-collection
        webcam_label.config(image=imgtk)

        if auto_submit:
            current_time = int(round(time.time() * 1000))
            if prev_frame is not None:
                change = detect_change(prev_frame, frame)
                if change > change_threshold:
                    change_detected = True
                    last_change_time = current_time
                elif change_detected and (current_time - last_change_time > stabilization_time):
                    change_detected = False
                    on_submit()
        prev_frame = frame
        webcam_label.after(100, update_image)

def detect_change(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    non_zero_count = np.sum(diff > change_sensitivity)
    return non_zero_count

def toggle_auto_submit():
    global auto_submit
    auto_submit = not auto_submit

root = tk.Tk(); root.title("Moondream"); root.geometry("800x580")
webcam_label = tk.Label(root); webcam_label.place(x=10, y=10, width=480, height=340) # Webcam image label
prompt_label = tk.Label(root, text="Prompt"); prompt_label.place(x=500, y=10)
prompt_entry = tk.Text(root, height=3, width=40, font=("Helvetica", 10), wrap=tk.WORD); prompt_entry.place(x=500, y=30)
submit_button = tk.Button(root, text="Submit", command=on_submit); submit_button.place(x=721, y=75)
output_label = tk.Label(root, text="Response"); output_label.place(x=500, y=95)
output_text = tk.Text(root, height=10, width=40, font=("Helvetica", 10), wrap=tk.WORD);output_text.place(x=500, y=115)
capture_button1 = tk.Button(root, text="1", command=lambda: capture_image(1)); capture_button1.place(x=650, y=350)
capture_button2 = tk.Button(root, text="2", command=lambda: capture_image(2)); capture_button2.place(x=675, y=350)
compare_prompt_label = tk.Label(root, text="Compare Prompt"); compare_prompt_label.place(x=500, y=265)
compare_prompt_entry = tk.Text(root, height=4, width=40, font=("Helvetica", 10), wrap=tk.WORD); compare_prompt_entry.place(x=500, y=285)
compare_button = tk.Button(root, text="Compare", command=on_compare);compare_button.place(x=710, y=345)
auto_submit_checkbox = tk.Checkbutton(root, text="Auto-Submit", var=tk.BooleanVar(value=True), command=toggle_auto_submit);auto_submit_checkbox.place(x=600, y=75)
image1_label = tk.Label(root, borderwidth=2, relief="groove");image1_label.place(x=10, y=365, width=240, height=180)
image2_label = tk.Label(root, borderwidth=2, relief="groove");image2_label.place(x=250, y=365, width=240, height=180)
stitched_image_label = tk.Label(root, borderwidth=2, relief="groove");stitched_image_label.place(x=500, y=355, width=120, height=195)

input_options = ["Webcam", "Screen Capture"]
selected_input = tk.StringVar()
selected_input.set("Webcam")  # Set the default option
input_dropdown = tk.OptionMenu(root, selected_input, *input_options)
input_dropdown.place(x=650, y=375)
input_dropdown.configure(font=("Helvetica", 10))

def on_mouse_drag(event):
    global last_pos, capture_area
    if last_pos is None:
        last_pos = (event.x, event.y)
        return
    dx = event.x - last_pos[0]
    dy = event.y - last_pos[1]
    capture_area['left'] -= dx
    capture_area['top'] -= dy
    if capture_area['left'] <0: capture_area['left']=0
    if capture_area["left"]+capture_area["width"] > screen_width: capture_area["left"]=screen_width-capture_area["width"]
    if capture_area['top'] <0: capture_area['top']=0    
    if capture_area["top"]+capture_area["height"] > screen_height: capture_area["top"]=screen_height-capture_area["height"]    
    last_pos = (event.x, event.y)

def on_mouse_release(event):
    global last_pos
    # Reset last_pos when the mouse is released
    last_pos = None


def on_mouse_wheel(event):
    global capture_area, zoom_factor
    # On Windows, use event.delta to determine the direction of scroll
    zoom_in = False
    #if event.delta > 0:
    #    zoom_in = True
    #elif event.delta < 0:
    #    zoom_in = False
    # On Linux, use event.num to determine the scroll direction
    if event.num == 4:
        zoom_in = True
    elif event.num == 5:
        zoom_in = False

    # Calculate new dimensions
    if zoom_in:
        capture_area['width'] = int(capture_area['width'] * zoom_factor)
        capture_area['height'] = int(capture_area['height'] * zoom_factor)
    else:
        capture_area['width'] = int(capture_area['width'] / zoom_factor)
        capture_area['height'] = int(capture_area['height'] / zoom_factor)

    if capture_area["width"]> screen_width: capture_area["width"]=screen_width
    if capture_area["height"]> screen_height: capture_area["height"]=screen_height
    if capture_area["left"]+capture_area["width"] > screen_width: capture_area["left"]=screen_width-capture_area["width"]  
    if capture_area["top"]+capture_area["height"] > screen_height: capture_area["top"]=screen_height-capture_area["height"]        


webcam_label.bind("<B1-Motion>", on_mouse_drag)
webcam_label.bind("<ButtonRelease-1>", on_mouse_release)
webcam_label.bind("<Button-4>", on_mouse_wheel)
webcam_label.bind("<Button-5>", on_mouse_wheel)

# Start capturing and updating the image
update_image()

# Start the GUI
root.mainloop()
